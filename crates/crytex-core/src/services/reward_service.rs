use std::sync::Arc;

use chrono::Utc;
use thiserror::Error;
use ulid::Ulid;

use crate::models::Experience;
use crate::persistence::{ExperienceRepository, PersistenceError};

/// Errors returned by the reward service.
#[derive(Debug, Error)]
pub enum RewardServiceError {
    #[error("persistence error: {0}")]
    Persistence(#[from] PersistenceError),
}

/// Parameters for [`RewardService::record`].
#[derive(Debug, Default)]
pub struct RecordRewardRequest<'a> {
    pub task_id: &'a str,
    pub project_id: Option<&'a str>,
    pub prompt_version_id: Option<&'a str>,
    pub critic_score: Option<f64>,
    pub human_score: Option<f64>,
    pub text: Option<&'a str>,
    pub comment: Option<&'a str>,
}

/// Computes and records reward from critic and human feedback scores.
pub struct RewardService {
    repo: Arc<dyn ExperienceRepository>,
}

impl RewardService {
    pub fn new(repo: Arc<dyn ExperienceRepository>) -> Self {
        Self { repo }
    }

    /// Compute reward as `0.6 * critic_score + 0.4 * human_score`.
    /// Missing scores are treated as 0.0.
    pub fn compute(critic_score: Option<f64>, human_score: Option<f64>) -> f64 {
        let critic = critic_score.unwrap_or(0.0);
        let human = human_score.unwrap_or(0.0);
        0.6 * critic + 0.4 * human
    }

    /// Record an experience and return the computed reward.
    pub async fn record(
        &self,
        request: RecordRewardRequest<'_>,
    ) -> Result<f64, RewardServiceError> {
        let reward = Self::compute(request.critic_score, request.human_score);
        let exp = Experience {
            id: Ulid::new().to_string(),
            task_id: request.task_id.to_string(),
            project_id: request.project_id.map(|s| s.to_string()),
            prompt_version_id: request.prompt_version_id.map(|s| s.to_string()),
            text: request.text.map(|s| s.to_string()),
            critic_score: request.critic_score,
            human_score: request.human_score,
            reward,
            comment: request.comment.map(|s| s.to_string()),
            created_at: Utc::now().timestamp_millis(),
        };
        self.repo.insert_experience(&exp).await?;
        Ok(reward)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use async_trait::async_trait;

    use crate::models::Experience;
    use crate::persistence::{ExperienceRepository, PersistenceError};

    use super::*;

    #[derive(Default)]
    struct MockExperienceRepo {
        experiences: Mutex<HashMap<String, Vec<Experience>>>,
    }

    #[async_trait]
    impl ExperienceRepository for MockExperienceRepo {
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

    #[test]
    fn reward_computes_blend() {
        let only_critic = RewardService::compute(Some(4.0), None);
        assert!((only_critic - 2.4).abs() < 0.001);

        let only_human = RewardService::compute(None, Some(5.0));
        assert!((only_human - 2.0).abs() < 0.001);

        let both = RewardService::compute(Some(4.0), Some(5.0));
        assert!((both - 4.4).abs() < 0.001);
    }

    #[tokio::test]
    async fn reward_service_persists_experience() {
        let repo = Arc::new(MockExperienceRepo::default());
        let service = RewardService::new(repo.clone());

        let reward = service
            .record(RecordRewardRequest {
                task_id: "t1",
                project_id: Some("p1"),
                prompt_version_id: Some("pv1"),
                critic_score: Some(4.0),
                human_score: Some(5.0),
                text: Some("input -> output"),
                comment: Some("good"),
            })
            .await
            .unwrap();

        assert!((reward - 4.4).abs() < 0.001);
        let experiences = repo.list_experiences_by_task("t1").await.unwrap();
        assert_eq!(experiences.len(), 1);
        assert_eq!(experiences[0].task_id, "t1");
        assert_eq!(experiences[0].project_id, Some("p1".to_string()));
        assert_eq!(experiences[0].prompt_version_id, Some("pv1".to_string()));
        assert_eq!(experiences[0].text, Some("input -> output".to_string()));
        assert_eq!(experiences[0].critic_score, Some(4.0));
        assert_eq!(experiences[0].human_score, Some(5.0));
        assert!((experiences[0].reward - 4.4).abs() < 0.001);
    }
}
