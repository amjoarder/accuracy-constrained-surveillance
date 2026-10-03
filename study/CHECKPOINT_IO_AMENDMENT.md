# Checkpoint I/O amendment on 3 October 2026

After an interruption, external evaluation resumed from241 completed records. A subsequent Windows reader/delete-sharing conflict raised PermissionError during atomic replacement of state.json at selected clip65. No completed clip record or experimental measurement was removed.

The atomic writer now retries transient PermissionError up to20 attempts with incremental50-ms delays, retaining the last valid checkpoint until replacement succeeds. This changes checkpoint reliability only, not prediction settings, image/frame selection, model weights, matching, metrics, selection criteria, or timing boundaries. All primary timed runs finished before this amendment.

test_checkpoint.py reproduced the failure with a Windows file handle denying delete sharing, then passed after the retry change. The three policy tests also passed. The external evaluation resumes by skipping completed per-clip JSON records and restarting only the unfinished clip.
