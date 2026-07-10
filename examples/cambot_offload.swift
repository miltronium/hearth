// cambot_offload.swift — a CAMBOT-style Swift consumer offloading work to HEARTH (Phase 5, G5).
//
// Mirrors examples/cambot_offload.py using the real `HearthClient` from
// swift/Sources/Hearth/HearthClient.swift. The offloaded call uses
// HearthOptions(allowEscalation: false) so HEARTH stays hard-local and the consumer's
// frontier budget is untouched.
//
// This is a documentation snippet, not part of the Swift package build. To run it, add the
// `Hearth` package as a dependency (see swift/README.md) and drop this into an executable
// target, or paste `offloadExample()` into an existing CAMBOT call site.
//
// Live wiring:
//   1. Start the daemon:   uv run hearth serve            (127.0.0.1:8080)
//   2. Read the token:     cat ~/.hearth/token
//   3. Pass baseURL + token below.
//   4. Measure savings:    uv run hearth stats --since 24h
//      (or GET /v1/hearth/admin/metrics — see docs/RUNBOOK_consumer_wiring.md)

import Foundation
import Hearth

/// Offload a summarize + classify subtask to a running local HEARTH gateway.
func offloadExample() async throws {
    // Token is written to ~/.hearth/token on first `hearth serve`.
    let token = ProcessInfo.processInfo.environment["HEARTH_TOKEN"]
    let hearth = HearthClient(
        baseURL: URL(string: "http://127.0.0.1:8080")!,
        token: token
    )

    let log = """
    The nightly build finished in 12m4s. Unit tests: 1,204 passed, 0 failed. \
    The linter flagged 3 style warnings in the payments module, all auto-fixable. \
    No regressions were detected.
    """

    // Hard-local: `summarize`/`classify` set allowEscalation=false internally, so these
    // spend 0 frontier tokens.
    let summary = try await hearth.summarize(text: log, maxWords: 25)
    let status = try await hearth.classify(text: log, labels: ["healthy", "degraded", "failing"])

    print("SUMMARY: \(summary)")
    print("STATUS : \(status)")
}
