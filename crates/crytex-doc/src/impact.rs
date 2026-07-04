//! Impact analysis over a [`CodeGraph`].

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::algo::tarjan_scc;
use petgraph::stable_graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::graph::{CodeGraph, EdgeKind, SymbolId, SymbolKind, SymbolNode};

/// Report produced by an impact analysis run.
#[derive(Debug, Clone, Default)]
pub struct ImpactReport {
    /// Symbols reachable from the starting set, ordered by graph distance.
    pub impacted: Vec<ImpactedSymbol>,
    /// Strongly connected components that contain at least one impacted symbol.
    pub cycles: Vec<Vec<SymbolId>>,
}

/// One symbol impacted by a change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactedSymbol {
    pub symbol: SymbolNode,
    /// Minimum number of reverse call-graph hops from the changed symbol.
    pub distance: usize,
    /// Number of distinct paths found.
    pub path_count: usize,
}

/// Analyzes the impact of changing one or more symbols.
pub struct ImpactAnalyzer<'g> {
    graph: &'g CodeGraph,
    sccs: Vec<Vec<NodeIndex>>,
}

impl<'g> ImpactAnalyzer<'g> {
    pub fn new(graph: &'g CodeGraph) -> Self {
        let sccs = tarjan_scc(&graph.graph);
        Self { graph, sccs }
    }

    /// Return all transitive callers of `symbol_id` up to `max_depth` hops.
    pub fn transitive_callers(
        &self,
        symbol_id: &SymbolId,
        max_depth: usize,
    ) -> Vec<ImpactedSymbol> {
        let start = match self.graph.index.get(symbol_id) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        let mut distances: HashMap<NodeIndex, (usize, usize)> = HashMap::new();

        visited.insert(start);
        queue.push_back((start, 0));
        distances.insert(start, (0, 1));

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for edge in self
                .graph
                .graph
                .edges_directed(node, petgraph::Direction::Incoming)
            {
                let relation = edge.weight();
                if relation.kind != EdgeKind::Calls {
                    continue;
                }
                let predecessor = edge.source();
                let new_depth = depth + 1;
                let entry = distances.entry(predecessor).or_insert((usize::MAX, 0));
                if new_depth < entry.0 {
                    entry.0 = new_depth;
                    entry.1 = 1;
                    queue.push_back((predecessor, new_depth));
                } else if new_depth == entry.0 {
                    entry.1 += 1;
                }
                visited.insert(predecessor);
            }
        }

        let mut impacted: Vec<ImpactedSymbol> = visited
            .into_iter()
            .filter(|&n| n != start)
            .map(|n| {
                let (distance, path_count) = distances[&n];
                ImpactedSymbol {
                    symbol: self.graph.graph[n].clone(),
                    distance,
                    path_count,
                }
            })
            .collect();

        impacted.sort_by_key(|i| (i.distance, i.symbol.name.clone()));
        impacted
    }

    /// Full impact set for a set of changed symbols.
    ///
    /// Includes transitive callers, implementors of changed traits/interfaces,
    /// and overriding methods. Returns symbols ranked by distance.
    pub fn impact_set(
        &self,
        start_ids: &[SymbolId],
        max_depth: usize,
        include_implementors: bool,
    ) -> ImpactReport {
        let mut impacted: Vec<ImpactedSymbol> = Vec::new();
        let mut impacted_ids: HashSet<SymbolId> = HashSet::new();
        let mut cycles: Vec<Vec<SymbolId>> = Vec::new();

        for id in start_ids {
            let callers = self.transitive_callers(id, max_depth);
            for c in callers {
                if impacted_ids.insert(c.symbol.id.clone()) {
                    impacted.push(c);
                }
            }

            if include_implementors && let Some(&idx) = self.graph.index.get(id) {
                for edge in self
                    .graph
                    .graph
                    .edges_directed(idx, petgraph::Direction::Incoming)
                {
                    if edge.weight().kind == EdgeKind::Implements {
                        let impl_idx = edge.source();
                        let symbol = self.graph.graph[impl_idx].clone();
                        if impacted_ids.insert(symbol.id.clone()) {
                            impacted.push(ImpactedSymbol {
                                symbol,
                                distance: 1,
                                path_count: 1,
                            });
                        }
                    }
                }
            }
        }

        impacted.sort_by_key(|i| (i.distance, i.symbol.name.clone()));

        // Report SCCs that contain any impacted symbol.
        let impacted_indices: HashSet<NodeIndex> = impacted
            .iter()
            .filter_map(|i| self.graph.index.get(&i.symbol.id).copied())
            .collect();
        for scc in &self.sccs {
            if scc.len() > 1 && scc.iter().any(|n| impacted_indices.contains(n)) {
                let ids: Vec<SymbolId> = scc
                    .iter()
                    .map(|&n| self.graph.graph[n].id.clone())
                    .collect();
                cycles.push(ids);
            }
        }

        ImpactReport { impacted, cycles }
    }

    /// Find the strongly connected component that contains `symbol_id`, if any.
    pub fn cycle_for(&self, symbol_id: &SymbolId) -> Option<Vec<SymbolNode>> {
        let idx = *self.graph.index.get(symbol_id)?;
        self.sccs
            .iter()
            .find(|scc| scc.contains(&idx) && scc.len() > 1)
            .map(|scc| scc.iter().map(|&n| self.graph.graph[n].clone()).collect())
    }

    /// All symbols in the graph that are public (best-effort).
    pub fn public_api(&self) -> Vec<&SymbolNode> {
        self.graph
            .symbols()
            .filter(|s| {
                s.visibility
                    .as_ref()
                    .map(|v| v == "pub" || v == "public")
                    .unwrap_or(false)
                    && matches!(
                        s.kind,
                        SymbolKind::Function
                            | SymbolKind::Method
                            | SymbolKind::Struct
                            | SymbolKind::Class
                    )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::languages::LanguageExtractor;
    use crate::graph::languages::rust::RustExtractor;

    fn rust_graph(source: &str) -> CodeGraph {
        let mut graph = CodeGraph::new();
        RustExtractor.extract("src/lib.rs", source, &mut graph);
        graph
    }

    #[test]
    fn impact_analysis_finds_callers_of_function() {
        let source = r#"
fn helper() {}
fn caller_a() { helper(); }
fn caller_b() { caller_a(); }
fn unused() {}
"#;
        let graph = rust_graph(source);
        let helper = graph
            .symbols()
            .find(|s| s.name == "helper" && s.kind == SymbolKind::Function)
            .unwrap();

        let analyzer = ImpactAnalyzer::new(&graph);
        let callers = analyzer.transitive_callers(&helper.id, 10);

        let names: Vec<String> = callers.iter().map(|c| c.symbol.name.clone()).collect();
        assert!(names.contains(&"caller_a".to_string()));
        assert!(names.contains(&"caller_b".to_string()));
        assert!(!names.contains(&"unused".to_string()));
    }

    #[test]
    fn impact_analysis_respects_max_depth() {
        let source = r#"
fn helper() {}
fn caller_a() { helper(); }
fn caller_b() { caller_a(); }
"#;
        let graph = rust_graph(source);
        let helper = graph
            .symbols()
            .find(|s| s.name == "helper" && s.kind == SymbolKind::Function)
            .unwrap();

        let analyzer = ImpactAnalyzer::new(&graph);
        let callers = analyzer.transitive_callers(&helper.id, 1);
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].symbol.name, "caller_a");
    }
}
