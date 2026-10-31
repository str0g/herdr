//! The clipboard Herdr shares with programs running inside panes.
//!
//! Pane children copy with OSC 52 writes and read the clipboard back with an OSC 52
//! query. Which clipboard those requests reach is a `[clipboard] mode` setting:
//!
//! - `local`: copies are forwarded to the attached client, and queries are answered
//!   from this one-slot cache of the newest copy Herdr forwarded. A yank in a pane
//!   can be pasted again by a clipboard-aware program (neovim, `xclip -o`) without a
//!   round trip to a real clipboard. Only the newest entry is retained: there is no
//!   clipboard history.
//! - `remote`: copies are written to the clipboard of the machine running the Herdr
//!   server and queries read that clipboard back through the platform helper.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::config::ClipboardModeConfig;

/// Largest entry served back to a querying child. libghostty-vt already refuses
/// larger OSC 52 writes, so this only bounds what a reply can carry.
const MAX_ENTRY_BYTES: usize = 192 * 1024;

const MODE_LOCAL: u8 = 0;
const MODE_REMOTE: u8 = 1;

fn mode_code(mode: ClipboardModeConfig) -> u8 {
    match mode {
        ClipboardModeConfig::Local => MODE_LOCAL,
        ClipboardModeConfig::Remote => MODE_REMOTE,
    }
}

/// One-slot clipboard shared by the panes of a Herdr server process.
///
/// Cheap to clone: the read callback installed on a pane terminal and the pane's
/// PTY reader hold the same slot, and every pane in the process shares it because
/// the attached client has one clipboard.
#[derive(Clone, Default)]
pub(crate) struct ClipboardCache {
    last_entry: Arc<Mutex<Option<Vec<u8>>>>,
    mode: Arc<AtomicU8>,
}

impl ClipboardCache {
    /// The cache shared by every pane terminal in this process.
    pub(crate) fn shared() -> Self {
        static SHARED: OnceLock<ClipboardCache> = OnceLock::new();
        SHARED.get_or_init(ClipboardCache::default).clone()
    }

    /// Records a clipboard write observed from a pane child. Oversized and empty
    /// writes are ignored so a child cannot grow the retained entry.
    pub(crate) fn record(&self, bytes: &[u8]) {
        if bytes.is_empty() || bytes.len() > MAX_ENTRY_BYTES {
            return;
        }
        let mut entry = self
            .last_entry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *entry = Some(bytes.to_vec());
    }

    /// The newest recorded entry, if any.
    pub(crate) fn last_entry(&self) -> Option<Vec<u8>> {
        self.last_entry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Which clipboard pane clipboard requests are served from.
    pub(crate) fn mode(&self) -> ClipboardModeConfig {
        match self.mode.load(Ordering::Acquire) {
            MODE_REMOTE => ClipboardModeConfig::Remote,
            _ => ClipboardModeConfig::Local,
        }
    }

    /// Applies the configured clipboard mode to every pane sharing this slot.
    pub(crate) fn set_mode(&self, mode: ClipboardModeConfig) {
        self.mode.store(mode_code(mode), Ordering::Release);
    }
}

/// Whether a read request asks for a text representation Herdr can serve.
/// Mirrors the MIME accepted on the write side, so a Kitty clipboard (OSC 5522)
/// request for another representation is refused instead of answered with text.
pub(crate) fn serves_requested_mime(mimes: &[&str]) -> bool {
    mimes.iter().any(|mime| {
        mime.split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .eq_ignore_ascii_case("text/plain")
    })
}

/// Answers a clipboard read request from the clipboard `[clipboard] mode` selects.
///
/// `read_host_clipboard` reads the clipboard of the machine running Herdr and runs
/// only in `remote` mode, on the pane's PTY reader thread, so it must stay bounded
/// by the platform helper's own timeout. An unservable request still gets a reply
/// (an empty clipboard) instead of leaving the querying program waiting.
pub(crate) fn serve_read(
    cache: &ClipboardCache,
    request: &crate::ghostty::ClipboardReadRequest<'_>,
    read_host_clipboard: impl FnOnce() -> Option<String>,
) -> crate::ghostty::ClipboardReadReply {
    if request.location != crate::ghostty::ClipboardLocation::Standard
        || !serves_requested_mime(request.mimes)
    {
        return crate::ghostty::ClipboardReadReply::Unsupported;
    }
    let entry = match cache.mode() {
        ClipboardModeConfig::Local => cache.last_entry(),
        ClipboardModeConfig::Remote => read_host_clipboard().map(|text| text.into_bytes()),
    };
    match entry {
        Some(entry) if !entry.is_empty() => crate::ghostty::ClipboardReadReply::Text(entry),
        Some(_) | None => crate::ghostty::ClipboardReadReply::Unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ghostty::{ClipboardLocation, ClipboardReadReply, ClipboardReadRequest};

    #[test]
    fn last_entry_tracks_the_newest_write() {
        let cache = ClipboardCache::default();
        assert_eq!(cache.last_entry(), None);

        cache.record(b"first");
        assert_eq!(cache.last_entry(), Some(b"first".to_vec()));
        cache.record(b"second");
        assert_eq!(cache.last_entry(), Some(b"second".to_vec()));
    }

    #[test]
    fn empty_and_oversized_writes_do_not_replace_the_entry() {
        let cache = ClipboardCache::default();
        cache.record(b"kept");

        cache.record(&[]);
        cache.record(&[b'x'; MAX_ENTRY_BYTES + 1]);

        assert_eq!(cache.last_entry(), Some(b"kept".to_vec()));
    }

    #[test]
    fn clones_share_the_same_slot() {
        let cache = ClipboardCache::default();
        cache.record(b"shared");

        let reader = cache.clone();
        reader.record(b"from another pane");
        assert_eq!(cache.last_entry(), Some(b"from another pane".to_vec()));
    }

    #[test]
    fn only_text_representations_are_served() {
        assert!(serves_requested_mime(&["text/plain"]));
        assert!(serves_requested_mime(&["text/plain;charset=utf-8"]));
        assert!(serves_requested_mime(&["image/png", "text/plain"]));
        assert!(!serves_requested_mime(&[]));
        assert!(!serves_requested_mime(&["image/png"]));
        assert!(!serves_requested_mime(&["UTF8_STRING"]));
    }

    #[test]
    fn clipboard_mode_defaults_to_local_and_spreads_to_clones() {
        let cache = ClipboardCache::default();
        assert_eq!(cache.mode(), ClipboardModeConfig::Local);

        let reader = cache.clone();
        cache.set_mode(ClipboardModeConfig::Remote);
        assert_eq!(reader.mode(), ClipboardModeConfig::Remote);

        cache.set_mode(ClipboardModeConfig::Local);
        assert_eq!(reader.mode(), ClipboardModeConfig::Local);
    }

    #[test]
    fn local_mode_answers_from_the_cache_without_touching_the_host() {
        let cache = ClipboardCache::default();
        cache.record(b"pane copy");

        let reply = serve_read(&cache, &text_read_request(), || {
            panic!("local mode must not read the host clipboard")
        });

        assert!(matches!(reply, ClipboardReadReply::Text(bytes) if bytes == b"pane copy"));
    }

    #[test]
    fn remote_mode_answers_from_the_host_clipboard_instead_of_the_cache() {
        let cache = ClipboardCache::default();
        cache.record(b"pane copy");
        cache.set_mode(ClipboardModeConfig::Remote);

        let reply = serve_read(&cache, &text_read_request(), || {
            Some("host clipboard".to_string())
        });

        assert!(matches!(reply, ClipboardReadReply::Text(bytes) if bytes == b"host clipboard"));
    }

    #[test]
    fn empty_or_missing_entries_are_answered_as_unsupported() {
        let cache = ClipboardCache::default();
        assert!(matches!(
            serve_read(&cache, &text_read_request(), || None),
            ClipboardReadReply::Unsupported
        ));

        cache.set_mode(ClipboardModeConfig::Remote);
        assert!(matches!(
            serve_read(&cache, &text_read_request(), || Some(String::new())),
            ClipboardReadReply::Unsupported
        ));
    }

    #[test]
    fn non_standard_or_non_text_requests_never_reach_the_host_clipboard() {
        let cache = ClipboardCache::default();
        cache.set_mode(ClipboardModeConfig::Remote);
        let request = crate::ghostty::ClipboardReadRequest {
            location: ClipboardLocation::Primary,
            mimes: &["text/plain"],
            list: false,
        };

        let reply = serve_read(&cache, &request, || {
            panic!("primary clipboard is not served")
        });

        assert!(matches!(reply, ClipboardReadReply::Unsupported));
    }

    fn text_read_request() -> ClipboardReadRequest<'static> {
        ClipboardReadRequest {
            location: ClipboardLocation::Standard,
            mimes: &["text/plain"],
            list: false,
        }
    }
}
