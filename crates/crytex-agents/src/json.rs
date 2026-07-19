use serde_json::Value;

pub(crate) fn strip_markdown_json_fence(content: &str) -> &str {
    let trimmed = content.trim();
    if let Some(inner) = trimmed.strip_prefix("```json") {
        inner.trim().trim_end_matches("```").trim()
    } else if let Some(inner) = trimmed.strip_prefix("```") {
        inner.trim().trim_end_matches("```").trim()
    } else {
        trimmed
    }
}

pub(crate) fn parse_llm_json_value(content: &str) -> Result<Value, serde_json::Error> {
    let cleaned = strip_markdown_json_fence(content);
    let without_trailing_commas = remove_trailing_json_commas(cleaned);
    let with_quoted_keys = quote_unquoted_json_object_keys(&without_trailing_commas);
    let balanced = close_unclosed_json_containers(&with_quoted_keys);
    serde_json::from_str(cleaned)
        .or_else(|_| serde_json::from_str(&without_trailing_commas))
        .or_else(|_| serde_json::from_str(&with_quoted_keys))
        .or_else(|_| serde_json::from_str(&balanced))
        .or_else(|_| serde_json::from_str(&insert_missing_json_commas_between_values(&balanced)))
}

fn remove_trailing_json_commas(input: &str) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(input.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];
        if in_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }

        if ch == '"' {
            in_string = true;
            output.push(ch);
            index += 1;
            continue;
        }

        if ch == ',' && next_non_whitespace_is_closing(&chars, index + 1) {
            index += 1;
            continue;
        }

        output.push(ch);
        index += 1;
    }

    output
}

fn next_non_whitespace_is_closing(chars: &[char], start: usize) -> bool {
    chars
        .iter()
        .skip(start)
        .find(|ch| !ch.is_whitespace())
        .is_some_and(|ch| matches!(ch, '}' | ']'))
}

fn insert_missing_json_commas_between_values(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut previous_significant = None;

    for ch in input.chars() {
        if in_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
                previous_significant = Some('"');
            }
            continue;
        }

        if ch == '"' {
            if previous_significant.is_some_and(value_can_precede_missing_comma) {
                output.push(',');
            }
            in_string = true;
            output.push(ch);
            continue;
        }

        output.push(ch);
        if !ch.is_whitespace() {
            previous_significant = Some(ch);
        }
    }

    output
}

fn value_can_precede_missing_comma(ch: char) -> bool {
    ch == '"' || ch == ']' || ch == '}' || ch.is_ascii_digit() || matches!(ch, 'e' | 'l')
}

fn quote_unquoted_json_object_keys(input: &str) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(input.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];
        if in_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }

        if ch == '"' {
            in_string = true;
            output.push(ch);
            index += 1;
            continue;
        }

        if matches!(ch, '{' | ',')
            && let Some((replacement, next_index)) = quoted_key_after_separator(&chars, index)
        {
            output.push(ch);
            output.push_str(&replacement);
            index = next_index;
            continue;
        }

        output.push(ch);
        index += 1;
    }

    output
}

fn quoted_key_after_separator(chars: &[char], separator_index: usize) -> Option<(String, usize)> {
    let mut cursor = separator_index + 1;
    let mut whitespace = String::new();
    while cursor < chars.len() && chars[cursor].is_whitespace() {
        whitespace.push(chars[cursor]);
        cursor += 1;
    }
    if cursor >= chars.len() || !is_jsonish_key_start(chars[cursor]) {
        return None;
    }

    let key_start = cursor;
    cursor += 1;
    while cursor < chars.len() && is_jsonish_key_continue(chars[cursor]) {
        cursor += 1;
    }
    let key = chars[key_start..cursor].iter().collect::<String>();

    let mut after_key = cursor;
    while after_key < chars.len() && chars[after_key].is_whitespace() {
        after_key += 1;
    }
    if chars.get(after_key) != Some(&':') {
        return None;
    }

    Some((format!("{whitespace}\"{key}\""), cursor))
}

fn is_jsonish_key_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_jsonish_key_continue(ch: char) -> bool {
    ch == '_' || ch == '-' || ch.is_ascii_alphanumeric()
}

fn close_unclosed_json_containers(input: &str) -> String {
    let mut output = String::with_capacity(input.len() + 8);
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for ch in input.chars() {
        output.push(ch);
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' if stack.last() == Some(&ch) => {
                stack.pop();
            }
            _ => {}
        }
    }

    if in_string {
        output.push('"');
    }
    for closing in stack.iter().rev() {
        output.push(*closing);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llm_json_value_accepts_trailing_object_comma() {
        let value =
            parse_llm_json_value(r#"{"summary":"ok","files_changed":[],"test_results":null,}"#)
                .unwrap();

        assert_eq!(value["summary"], "ok");
    }

    #[test]
    fn parse_llm_json_value_accepts_nested_trailing_array_comma() {
        let value = parse_llm_json_value(
            r#"{"blocking_issues":[{"reason":"missing",},],"review_decision":"reject",}"#,
        )
        .unwrap();

        assert_eq!(value["blocking_issues"][0]["reason"], "missing");
    }

    #[test]
    fn parse_llm_json_value_preserves_commas_inside_strings() {
        let value =
            parse_llm_json_value(r#"{"summary":"keep comma, } text","items":["a, b",],}"#).unwrap();

        assert_eq!(value["summary"], "keep comma, } text");
        assert_eq!(value["items"][0], "a, b");
    }

    #[test]
    fn parse_llm_json_value_accepts_missing_comma_between_fields() {
        let value = parse_llm_json_value(
            r#"{
                "summary": "ok"
                "files_changed": []
            }"#,
        )
        .unwrap();

        assert_eq!(value["summary"], "ok");
        assert_eq!(value["files_changed"], serde_json::json!([]));
    }

    #[test]
    fn parse_llm_json_value_accepts_unquoted_object_keys() {
        let value = parse_llm_json_value(
            r#"{
                files_changed: [{"path":"src/lib.rs","action":"modified"}],
                test_results: null,
                summary: "ok"
            }"#,
        )
        .unwrap();

        assert_eq!(value["summary"], "ok");
        assert_eq!(value["files_changed"][0]["path"], "src/lib.rs");
    }

    #[test]
    fn parse_llm_json_value_accepts_unclosed_object_suffix() {
        let value = parse_llm_json_value(
            r#"{"plan":{"goal":"ship","assumptions":[],"subtasks":[{"kind":"codegen"}]"#,
        )
        .unwrap();

        assert_eq!(value["plan"]["goal"], "ship");
        assert_eq!(value["plan"]["subtasks"][0]["kind"], "codegen");
    }
}
