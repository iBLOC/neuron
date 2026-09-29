//! Cross-platform process RSS / total-RAM probe for the model
//! scheduler.
//!
//! The scheduler uses two numbers:
//!
//! * `rss_bytes()` — the resident-set delta across a model unload is
//!   the *observed* footprint fed back into the memory profile store
//!   (EWMA), which is how the shipped reference estimates converge on
//!   the actual runtime platform over time.
//! * `total_memory_bytes()` — derives the automatic memory budget
//!   when `NEURON_SCHEDULER_BUDGET_MB` is unset.
//!
//! Mirrors the measurement code in `benches/memory.rs` (Linux
//! `/proc`, macOS `task_info`, Windows `K32GetProcessMemoryInfo`) but
//! trimmed to what the scheduler needs at runtime. Every accessor
//! returns 0 on failure rather than panicking — a missing probe must
//! degrade scheduling quality, never crash serving.

/// Current process resident set size in bytes (0 when unavailable).
pub fn rss_bytes() -> u64 {
    let mut s = Snapshot::default();
    platform::read(&mut s);
    s.rss_bytes
}

/// Total physical RAM in bytes (0 when unavailable).
pub fn total_memory_bytes() -> u64 {
    platform::total()
}

#[derive(Default, Clone, Copy)]
struct Snapshot {
    rss_bytes: u64,
}

#[cfg(target_os = "linux")]
mod platform {
    use super::Snapshot;

    pub fn read(s: &mut Snapshot) {
        if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                let mut parts = line.split_whitespace();
                if parts.next() == Some("VmRSS:")
                    && let Some(kb) = parts.next().and_then(|v| v.parse::<u64>().ok())
                {
                    s.rss_bytes = kb * 1024;
                }
            }
        }
    }

    pub fn total() -> u64 {
        // MemTotal is reported in kB.
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                let mut parts = line.split_whitespace();
                if parts.next() == Some("MemTotal:")
                    && let Some(kb) = parts.next().and_then(|v| v.parse::<u64>().ok())
                {
                    return kb * 1024;
                }
            }
        }
        0
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::Snapshot;

    pub fn read(s: &mut Snapshot) {
        use libc::{mach_msg_type_number_t, mach_task_self_, task_vm_info, TASK_VM_INFO};
        unsafe {
            let mut info: task_vm_info = std::mem::zeroed();
            let count =
                (std::mem::size_of::<task_vm_info>() / std::mem::size_of::<libc::natural_t>())
                    as mach_msg_type_number_t;
            let kr = libc::task_info(
                mach_task_self_ as libc::task_t,
                TASK_VM_INFO,
                (&mut info as *mut _) as libc::task_info_t,
                &count as *mut _ as *mut mach_msg_type_number_t,
            );
            if kr == libc::KERN_SUCCESS {
                s.rss_bytes = info.resident_size as u64;
            }
        }
    }

    pub fn total() -> u64 {
        unsafe {
            let mut size = std::mem::size_of::<u64>();
            let mut value: u64 = 0;
            let name = b"hw.memsize\0";
            if libc::sysctlbyname(
                name.as_ptr() as *const libc::c_char,
                (&mut value as *mut u64).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            ) == 0
            {
                return value;
            }
        }
        0
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use super::Snapshot;

    pub fn read(s: &mut Snapshot) {
        use windows_sys::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        unsafe {
            let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            let cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            if K32GetProcessMemoryInfo(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                &mut counters,
                cb,
            ) != 0
            {
                s.rss_bytes = counters.WorkingSetSize as u64;
            }
        }
    }

    pub fn total() -> u64 {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        unsafe {
            let mut status: MEMORYSTATUSEX = std::mem::zeroed();
            status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut status) != 0 {
                return status.ullTotalPhys as u64;
            }
        }
        0
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod platform {
    use super::Snapshot;

    pub fn read(_s: &mut Snapshot) {}
    pub fn total() -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rss_probe_returns_nonzero_on_supported_platforms() {
        let rss = rss_bytes();
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
        assert!(rss > 0, "RSS probe should work on this platform");
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        assert_eq!(rss, 0);
    }

    #[test]
    fn test_total_memory_probe_returns_nonzero_on_supported_platforms() {
        let total = total_memory_bytes();
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
        assert!(total > 0, "total-RAM probe should work on this platform");
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        assert_eq!(total, 0);
    }
}
