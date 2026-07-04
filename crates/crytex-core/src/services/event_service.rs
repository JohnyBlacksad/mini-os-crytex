use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::bus::{Event, EventBus};

/// Handler for events emitted by the kernel.
#[async_trait]
pub trait EventHandler: Send + Sync {
    async fn handle(&self, event: Event);
}

/// Service interface for publishing and subscribing to domain events.
#[async_trait]
pub trait EventService: Send + Sync {
    /// Publish an event to all subscribers.
    fn publish(&self, event: Event);

    /// Subscribe to future events.
    fn subscribe(&self) -> broadcast::Receiver<Event>;

    /// Start a handler task that processes events asynchronously.
    async fn start_handler(&self, handler: Arc<dyn EventHandler>);
}

/// Default implementation of [`EventService`] wrapping [`EventBus`].
pub struct EventServiceImpl {
    bus: Arc<EventBus>,
}

impl EventServiceImpl {
    pub fn new(bus: Arc<EventBus>) -> Self {
        Self { bus }
    }
}

#[async_trait]
impl EventService for EventServiceImpl {
    fn publish(&self, event: Event) {
        self.bus.publish(event);
    }

    fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.bus.subscribe()
    }

    async fn start_handler(&self, handler: Arc<dyn EventHandler>) {
        let mut rx = self.subscribe();
        tokio::spawn(async move {
            while let Ok(event) = rx.recv().await {
                handler.handle(event).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn publish_delivers_to_subscriber() {
        let bus = Arc::new(EventBus::new());
        let svc = EventServiceImpl::new(bus);
        let mut rx = svc.subscribe();

        svc.publish(Event::TaskStarted {
            task_id: "t1".into(),
        });

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, Event::TaskStarted { .. }));
    }

    #[tokio::test]
    async fn handler_receives_events() {
        let bus = Arc::new(EventBus::new());
        let svc = EventServiceImpl::new(bus);

        struct CountingHandler {
            count: AtomicUsize,
        }

        #[async_trait]
        impl EventHandler for CountingHandler {
            async fn handle(&self, _event: Event) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }

        let handler = Arc::new(CountingHandler {
            count: AtomicUsize::new(0),
        });

        svc.start_handler(handler.clone()).await;
        svc.publish(Event::TaskStarted {
            task_id: "t1".into(),
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(handler.count.load(Ordering::SeqCst), 1);
    }
}
