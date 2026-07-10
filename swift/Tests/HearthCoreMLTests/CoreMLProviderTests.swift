// Tests for the Phase 6 Core ML on-device path: the `CoreMLProvider` conformer to
// `HearthInference`. These stay offline and need no real model file — where Core ML isn't
// present (or no model is exported) we assert the *unavailable* contract instead.

import XCTest
import Hearth
@testable import HearthCoreML

#if canImport(CoreML)
import CoreML
#endif

final class CoreMLProviderTests: XCTestCase {

    /// A URL that cannot resolve to a loadable model, on any platform.
    private var bogusModelURL: URL {
        URL(fileURLWithPath: "/nonexistent/hearth-tests/does-not-exist.mlpackage")
    }

    // MARK: Availability contract (isAvailable / unavailableReason agree)

    func testProviderStaticAvailabilityIsConsistent() {
        if CoreMLProvider.isAvailable {
            // Core ML present: no build-time reason. (Per-URL model availability is separate.)
            XCTAssertNil(CoreMLProvider.unavailableReason)
        } else {
            XCTAssertNotNil(CoreMLProvider.unavailableReason)
        }
    }

    // MARK: Construction from a bogus URL throws a clear error, on any toolchain

    func testInitFromBogusURLThrowsOnDeviceUnavailable() {
        XCTAssertThrowsError(try CoreMLProvider(modelURL: bogusModelURL)) { error in
            guard case HearthError.onDeviceUnavailable(let reason) = error else {
                return XCTFail("expected .onDeviceUnavailable, got \(error)")
            }
            XCTAssertFalse(reason.isEmpty)
        }
    }

    // MARK: Protocol conformance (compile-time proof it's swappable behind the interface)

    func testCoreMLProviderConformsToHearthInference() {
        // Constructing throws (bogus URL / no framework), but the type must satisfy the
        // protocol so it's usable as `any HearthInference` at call sites.
        func acceptsInference(_ type: (any HearthInference.Type)) { _ = type }
        acceptsInference(CoreMLProvider.self)
    }

    // MARK: Stub-build guarantee — when Core ML can't be imported, everything is unavailable

    #if !canImport(CoreML)
    func testStubReportsUnavailable() {
        XCTAssertFalse(CoreMLProvider.isAvailable)
        XCTAssertNotNil(CoreMLProvider.unavailableReason)
    }
    #endif
}
