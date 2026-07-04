use serde::{Deserialize, Serialize};

/// Capabilities that a tool may require.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability(u32);

impl Capability {
    pub const READ: Capability = Capability(1 << 0);
    pub const WRITE: Capability = Capability(1 << 1);
    pub const SHELL: Capability = Capability(1 << 2);
    pub const NETWORK: Capability = Capability(1 << 3);
    pub const GIT: Capability = Capability(1 << 4);

    pub const fn empty() -> Self {
        Capability(0)
    }

    pub const fn all() -> Self {
        Capability(Self::READ.0 | Self::WRITE.0 | Self::SHELL.0 | Self::NETWORK.0 | Self::GIT.0)
    }

    pub fn contains(&self, other: Capability) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn union(&self, other: Capability) -> Capability {
        Capability(self.0 | other.0)
    }
}

/// Human-readable permission set used at runtime.
pub type PermissionSet = Capability;
