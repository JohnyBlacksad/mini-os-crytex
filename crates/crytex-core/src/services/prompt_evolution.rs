//! Prompt evolution service.
//!
//! Manages a population of system prompt versions per agent, computes fitness
//! from recorded experiences, and applies deterministic mutation operators.

use std::sync::Arc;

use chrono::Utc;
use rand::RngCore;
use thiserror::Error;
use ulid::Ulid;

use crate::models::PromptVersion;
use crate::persistence::{ExperienceRepository, PersistenceError, PromptVersionRepository};

/// Errors returned by [`PromptEvolutionService`].
#[derive(Debug, Error)]
pub enum PromptEvolutionError {
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
    #[error("prompt version not found: {0}")]
    VersionNotFound(String),
    #[error("no population for agent: {0}")]
    EmptyPopulation(String),
}

/// Deterministic mutation operators for a system prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationOperator {
    /// Rephrase the final instruction block.
    Rephrase,
    /// Append an extra constraint.
    AddConstraint,
    /// Inject a short JSON example.
    InjectExample,
    /// Change the tone directive.
    ChangeTone,
}

impl MutationOperator {
    /// All supported operators, in a stable order.
    pub fn all() -> &'static [MutationOperator] {
        &[
            MutationOperator::Rephrase,
            MutationOperator::AddConstraint,
            MutationOperator::InjectExample,
            MutationOperator::ChangeTone,
        ]
    }

    /// Apply the operator to a prompt string.
    pub fn apply(&self, prompt: &str) -> String {
        match self {
            MutationOperator::Rephrase => format!(
                "{}\n\n[Rephrased instruction] Express the same requirements using different wording while preserving every rule above.",
                prompt
            ),
            MutationOperator::AddConstraint => format!(
                "{}\n\n[Added constraint] Every response must begin with a one-sentence summary of the intended outcome.",
                prompt
            ),
            MutationOperator::InjectExample => format!(
                "{}\n\n[Injected example] Example of a valid response snippet: {{ \"summary\": \"...\", \"result\": \"...\" }}.",
                prompt
            ),
            MutationOperator::ChangeTone => format!("[Tone: precise and concise]\n\n{}", prompt),
        }
    }
}

/// Evolves system prompts for agents based on reward feedback.
pub struct PromptEvolutionService<R, E> {
    prompt_repo: Arc<R>,
    experience_repo: Arc<E>,
}

impl<R, E> PromptEvolutionService<R, E>
where
    R: PromptVersionRepository,
    E: ExperienceRepository,
{
    pub fn new(prompt_repo: Arc<R>, experience_repo: Arc<E>) -> Self {
        Self {
            prompt_repo,
            experience_repo,
        }
    }

    /// Create the initial active version for an agent from a base prompt.
    pub async fn seed_agent(
        &self,
        agent: &str,
        base_prompt: &str,
    ) -> Result<PromptVersion, PromptEvolutionError> {
        let version = PromptVersion {
            id: Ulid::new().to_string(),
            agent: agent.to_string(),
            project_id: None,
            system_prompt: base_prompt.to_string(),
            fitness: None,
            parent_id: None,
            created_at: Utc::now().timestamp_millis(),
            active: true,
        };
        self.prompt_repo.insert_prompt_version(&version).await?;
        self.prompt_repo
            .set_active_prompt_version(&version.id, agent)
            .await?;
        Ok(version)
    }

    /// Create a mutated child of an existing prompt version.
    pub async fn mutate(
        &self,
        parent_id: &str,
        operator: MutationOperator,
    ) -> Result<PromptVersion, PromptEvolutionError> {
        let parent = self
            .prompt_repo
            .get_prompt_version(parent_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(parent_id.to_string()))?;

        let child = PromptVersion {
            id: Ulid::new().to_string(),
            agent: parent.agent.clone(),
            project_id: parent.project_id.clone(),
            system_prompt: operator.apply(&parent.system_prompt),
            fitness: None,
            parent_id: Some(parent_id.to_string()),
            created_at: Utc::now().timestamp_millis(),
            active: false,
        };

        self.prompt_repo.insert_prompt_version(&child).await?;
        Ok(child)
    }

    /// Run one evolution step: tournament select, mutate, persist.
    pub async fn evolve_step(
        &self,
        agent: &str,
        operator: MutationOperator,
        tournament_size: usize,
        rng: &mut impl RngCore,
    ) -> Result<PromptVersion, PromptEvolutionError> {
        let parent = self
            .tournament_select(agent, tournament_size, rng)
            .await?
            .ok_or_else(|| PromptEvolutionError::EmptyPopulation(agent.to_string()))?;
        self.mutate(&parent.id, operator).await
    }

    /// Tournament selection over versions with known fitness.
    /// Versions without fitness are treated as fitness 0.0.
    pub async fn tournament_select(
        &self,
        agent: &str,
        tournament_size: usize,
        rng: &mut impl RngCore,
    ) -> Result<Option<PromptVersion>, PromptEvolutionError> {
        let population = self
            .prompt_repo
            .list_prompt_versions_by_agent(agent)
            .await?;
        if population.is_empty() {
            return Ok(None);
        }

        let size = tournament_size.min(population.len()).max(1);
        let mut selected: Vec<&PromptVersion> = Vec::with_capacity(size);
        while selected.len() < size {
            let idx = (rng.next_u32() as usize) % population.len();
            let candidate = &population[idx];
            if !selected.contains(&candidate) {
                selected.push(candidate);
            }
        }

        Ok(selected
            .into_iter()
            .max_by(|a, b| {
                let af = a.fitness.unwrap_or(0.0);
                let bf = b.fitness.unwrap_or(0.0);
                af.partial_cmp(&bf).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned())
    }

    /// Recompute the average reward for a prompt version and persist it.
    pub async fn recompute_fitness(&self, version_id: &str) -> Result<f64, PromptEvolutionError> {
        let experiences = self
            .experience_repo
            .list_experiences_by_prompt_version(version_id)
            .await?;
        let fitness = if experiences.is_empty() {
            0.0
        } else {
            experiences.iter().map(|e| e.reward).sum::<f64>() / experiences.len() as f64
        };

        let mut version = self
            .prompt_repo
            .get_prompt_version(version_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(version_id.to_string()))?;
        version.fitness = Some(fitness);
        self.prompt_repo.update_prompt_version(&version).await?;
        Ok(fitness)
    }

    /// Activate a prompt version and deactivate all other versions for the same agent.
    pub async fn activate(&self, version_id: &str) -> Result<(), PromptEvolutionError> {
        let version = self
            .prompt_repo
            .get_prompt_version(version_id)
            .await?
            .ok_or_else(|| PromptEvolutionError::VersionNotFound(version_id.to_string()))?;
        self.prompt_repo
            .set_active_prompt_version(version_id, &version.agent)
            .await?;
        Ok(())
    }

    /// Return the active version for an agent, if any.
    pub async fn active_version(
        &self,
        agent: &str,
    ) -> Result<Option<PromptVersion>, PromptEvolutionError> {
        Ok(self.prompt_repo.get_active_prompt_version(agent).await?)
    }

    /// Return the system prompt of the active version for an agent.
    pub async fn system_prompt_for_agent(
        &self,
        agent: &str,
    ) -> Result<Option<String>, PromptEvolutionError> {
        Ok(self.active_version(agent).await?.map(|v| v.system_prompt))
    }

    /// List all versions for an agent.
    pub async fn list_versions(
        &self,
        agent: &str,
    ) -> Result<Vec<PromptVersion>, PromptEvolutionError> {
        Ok(self
            .prompt_repo
            .list_prompt_versions_by_agent(agent)
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::models::Experience;

    #[derive(Default)]
    struct TestRepo {
        versions: Mutex<HashMap<String, PromptVersion>>,
        experiences: Mutex<HashMap<String, Vec<Experience>>>,
    }

    #[async_trait]
    impl PromptVersionRepository for TestRepo {
        async fn insert_prompt_version(
            &self,
            version: &PromptVersion,
        ) -> Result<(), PersistenceError> {
            self.versions
                .lock()
                .unwrap()
                .insert(version.id.clone(), version.clone());
            Ok(())
        }
        async fn update_prompt_version(
            &self,
            version: &PromptVersion,
        ) -> Result<(), PersistenceError> {
            self.versions
                .lock()
                .unwrap()
                .insert(version.id.clone(), version.clone());
            Ok(())
        }
        async fn get_prompt_version(
            &self,
            id: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(self.versions.lock().unwrap().get(id).cloned())
        }
        async fn list_prompt_versions_by_agent(
            &self,
            agent: &str,
        ) -> Result<Vec<PromptVersion>, PersistenceError> {
            Ok(self
                .versions
                .lock()
                .unwrap()
                .values()
                .filter(|v| v.agent == agent)
                .cloned()
                .collect())
        }
        async fn get_active_prompt_version(
            &self,
            agent: &str,
        ) -> Result<Option<PromptVersion>, PersistenceError> {
            Ok(self
                .versions
                .lock()
                .unwrap()
                .values()
                .find(|v| v.agent == agent && v.active)
                .cloned())
        }
        async fn set_active_prompt_version(
            &self,
            id: &str,
            agent: &str,
        ) -> Result<(), PersistenceError> {
            let mut versions = self.versions.lock().unwrap();
            for v in versions.values_mut() {
                if v.agent == agent {
                    v.active = false;
                }
            }
            if let Some(v) = versions.get_mut(id) {
                v.active = true;
            }
            Ok(())
        }
    }

    #[async_trait]
    impl ExperienceRepository for TestRepo {
        async fn insert_experience(&self, exp: &Experience) -> Result<(), PersistenceError> {
            self.experiences
                .lock()
                .unwrap()
                .entry(exp.task_id.clone())
                .or_default()
                .push(exp.clone());
            Ok(())
        }
        async fn list_experiences_by_task(
            &self,
            task_id: &str,
        ) -> Result<Vec<Experience>, PersistenceError> {
            Ok(self
                .experiences
                .lock()
                .unwrap()
                .get(task_id)
                .cloned()
                .unwrap_or_default())
        }
        async fn list_experiences_by_prompt_version(
            &self,
            prompt_version_id: &str,
        ) -> Result<Vec<Experience>, PersistenceError> {
            Ok(self
                .experiences
                .lock()
                .unwrap()
                .values()
                .flat_map(|v| v.iter())
                .filter(|e| e.prompt_version_id.as_deref() == Some(prompt_version_id))
                .cloned()
                .collect())
        }
    }

    fn make_service() -> PromptEvolutionService<TestRepo, TestRepo> {
        let repo = Arc::new(TestRepo::default());
        PromptEvolutionService::new(repo.clone(), repo)
    }

    #[tokio::test]
    async fn seed_creates_initial_active_version() {
        let service = make_service();
        let v = service.seed_agent("coder", "base prompt").await.unwrap();

        assert_eq!(v.agent, "coder");
        assert_eq!(v.system_prompt, "base prompt");
        assert!(v.active);
        assert!(v.parent_id.is_none());

        let active = service.active_version("coder").await.unwrap();
        assert_eq!(active.map(|a| a.id), Some(v.id));
    }

    #[tokio::test]
    async fn mutate_produces_child_with_changed_text_and_parent() {
        let service = make_service();
        let parent = service.seed_agent("coder", "base prompt").await.unwrap();
        let child = service
            .mutate(&parent.id, MutationOperator::Rephrase)
            .await
            .unwrap();

        assert_eq!(child.parent_id, Some(parent.id));
        assert_eq!(child.agent, "coder");
        assert_ne!(child.system_prompt, parent.system_prompt);
        assert!(!child.active);
    }

    #[tokio::test]
    async fn tournament_select_returns_highest_fitness_in_sample() {
        let service = make_service();
        let base = service.seed_agent("coder", "base").await.unwrap();
        let mut versions = vec![base.clone()];
        for (i, op) in MutationOperator::all().iter().enumerate() {
            let child = service.mutate(&base.id, *op).await.unwrap();
            let mut child = child;
            child.fitness = Some((i + 1) as f64);
            service
                .prompt_repo
                .update_prompt_version(&child)
                .await
                .unwrap();
            versions.push(child);
        }

        // Set base fitness low and make the tournament include the whole population
        // so the winner is deterministically the globally fittest version.
        let mut base_low = base;
        base_low.fitness = Some(0.0);
        service
            .prompt_repo
            .update_prompt_version(&base_low)
            .await
            .unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let winner = service
            .tournament_select("coder", versions.len(), &mut rng)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(winner.fitness, Some(4.0));
    }

    #[tokio::test]
    async fn recompute_fitness_averages_rewards() {
        let service = make_service();
        let version = service.seed_agent("coder", "base").await.unwrap();

        for reward in [3.0, 5.0] {
            let exp = Experience {
                id: Ulid::new().to_string(),
                task_id: Ulid::new().to_string(),
                project_id: None,
                prompt_version_id: Some(version.id.clone()),
                text: None,
                critic_score: None,
                human_score: None,
                reward,
                comment: None,
                created_at: 0,
            };
            service
                .experience_repo
                .insert_experience(&exp)
                .await
                .unwrap();
        }

        let fitness = service.recompute_fitness(&version.id).await.unwrap();
        assert!((fitness - 4.0).abs() < 0.001);

        let updated = service
            .prompt_repo
            .get_prompt_version(&version.id)
            .await
            .unwrap()
            .unwrap();
        assert!((updated.fitness.unwrap() - 4.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn activate_makes_only_one_active_per_agent() {
        let service = make_service();
        let v1 = service.seed_agent("coder", "base").await.unwrap();
        let v2 = service
            .mutate(&v1.id, MutationOperator::ChangeTone)
            .await
            .unwrap();

        service.activate(&v2.id).await.unwrap();

        let active = service.active_version("coder").await.unwrap().unwrap();
        assert_eq!(active.id, v2.id);

        let v1_loaded = service
            .prompt_repo
            .get_prompt_version(&v1.id)
            .await
            .unwrap()
            .unwrap();
        assert!(!v1_loaded.active);
    }

    #[tokio::test]
    async fn system_prompt_for_agent_returns_active_override() {
        let service = make_service();
        service.seed_agent("coder", "active prompt").await.unwrap();
        let prompt = service.system_prompt_for_agent("coder").await.unwrap();
        assert_eq!(prompt, Some("active prompt".to_string()));
    }
}
