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
    serde_json::from_str(cleaned)
        .or_else(|_| serde_json::from_str(&without_trailing_commas))
        .or_else(|_| {
            serde_json::from_str(&insert_missing_json_commas_between_values(
                &without_trailing_commas,
            ))
        })
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
}
