(function () {
  "use strict";

  function cloneDraft(rows) {
    if (!Array.isArray(rows)) return null;
    return rows.map(function (row) {
      return {
        slice_index: row.slice_index,
        frame_role: row.frame_role,
        structure: row.structure,
        included: Boolean(row.included)
      };
    });
  }

  function lineageFromState(value) {
    if (!value) return null;
    return {
      head_revision_id: value.head_revision_id == null ? null : value.head_revision_id,
      selection_id: value.selection_id == null ? null : value.selection_id,
      inclusion_set_id: value.inclusion_set_id == null ? null : value.inclusion_set_id
    };
  }

  function cloneLineage(value) {
    return lineageFromState(value);
  }

  // Keep locally edited rows and the identities they were captured against
  // together. A heartbeat may refresh the projection, but never its lineage.
  function updateDraftLineage(previous, event) {
    const prior = previous || {draft: null, lineage: null, dirty: false};
    const serverState = event && event.state ? event.state : null;
    const serverDraft = cloneDraft(serverState && serverState.inclusion_draft);
    const serverLineage = lineageFromState(serverState);
    if (!event || typeof event.type !== "string") throw new Error("draft lineage event is required");
    if (event.type === "server_state") {
      if (prior.dirty) {
        return {
          draft: cloneDraft(prior.draft),
          lineage: cloneLineage(prior.lineage),
          dirty: true
        };
      }
      return {draft: serverDraft, lineage: serverLineage, dirty: false};
    }
    const eventDraft = Object.prototype.hasOwnProperty.call(event, "draft")
      ? cloneDraft(event.draft)
      : serverDraft;
    if (event.type === "edit") {
      return {
        draft: eventDraft,
        lineage: cloneLineage(prior.lineage) || serverLineage,
        dirty: true
      };
    }
    if (event.type === "commit" || event.type === "reset") {
      return {draft: eventDraft, lineage: serverLineage, dirty: event.type === "reset"};
    }
    throw new Error("unknown draft lineage event");
  }

  // --- S3-008B §3.2 canonical serialization (JS side) -----------------------
  // The SAME shared numeric LEXICAL GRAMMAR as the Python producer
  // (cmr_viewer.state.canonical_number_lex): shortest-roundtrip decimal, then
  // drop a trailing ".0". Python uses repr(float(v)) + strip; JS uses
  // String(Number(v)). Both recipes are byte-identical over the accepted
  // domain, which the §5.7 cross-runtime test pins.
  const STRUCTURE_ORDER = ["lv_cavity", "lv_myocardium", "rv_cavity"];
  const NUMBER_LEX_RE = /^(0|[1-9][0-9]*(?:\.[0-9]+)?)$/;

  function numberLex(value) {
    const token = String(Number(value));
    if (!NUMBER_LEX_RE.test(token)) {
      throw new Error("display number outside the frozen grammar: " + token);
    }
    return token;
  }

  function canonicalWindowLevelJson(windowLevel) {
    // Canonical compact SORTED-key JSON: {"center":40,"width":400}
    return (
      '{"center":' + numberLex(windowLevel.center) +
      ',"width":' + numberLex(windowLevel.width) + "}"
    );
  }

  function canonicalVisibleStructuresJson(visible) {
    // Order is the frozen Structure enum order.
    return "[" + visible.map(function (name) {
      return JSON.stringify(name);
    }).join(",") + "]";
  }

  window.CMR_WORKSTATION = {
    updateDraftLineage: updateDraftLineage,
    STRUCTURE_ORDER: STRUCTURE_ORDER,
    numberLex: numberLex,
    canonicalWindowLevelJson: canonicalWindowLevelJson,
    canonicalVisibleStructuresJson: canonicalVisibleStructuresJson
  };
  if (!window.document || typeof window.document.getElementById !== "function") return;

  const canvas = document.getElementById("cine-canvas");
  const context = canvas.getContext("2d");
  const message = document.getElementById("message");
  const grid = document.getElementById("inclusion-grid");
  let seriesUid = null;
  let state = null;
  let bundle = null;
  let bundleRows = [];
  let bundlePngs = Object.create(null);
  let playing = false;
  let frame = 0;
  let slice = 0;
  let fps = 10;
  let edFrame = null;
  let esFrame = null;
  let zoom = 1;
  let bundleContext = "";
  let frameRequestCount = 0;
  let draftLineage = {draft: null, lineage: null, dirty: false};
  // --- S3-008B display-only view context (§3.1/§3.3) -----------------------
  // Request-scoped and NEVER an action field, NEVER persisted. windowLevel is
  // null (per-frame auto range) or {center, width}; visibleStructures is null
  // (ALL structures), [] (no contours), or Structure names in enum order.
  let windowLevel = null;
  let visibleStructures = null;

  function show(text, error) {
    message.textContent = text;
    message.classList.toggle("error", Boolean(error));
  }

  function syncDraftProjection() {
    if (state && draftLineage.draft !== null) {
      state.inclusion_draft = cloneDraft(draftLineage.draft);
    }
  }

  function queryView(includeFrame) {
    const params = new URLSearchParams();
    params.set("slice", String(slice));
    if (includeFrame) params.set("frame", String(frame));
    params.set("fps", String(fps));
    params.set("playing", playing ? "true" : "false");
    const sex = document.getElementById("reference-sex").value;
    if (sex) params.set("reference_sex", sex);
    params.set("inclusion_draft", JSON.stringify(currentDraft()));
    // S3-008B §3.1/§3.2: two OPTIONAL display-only params. Omission is the
    // canonical default, so null contributes NO parameter at all. The value is
    // canonical compact sorted-key JSON percent-encoded with
    // encodeURIComponent (byte-identical to Python quote(safe="") here).
    // URLSearchParams.toString() applies the SAME percent-encoding the
    // Python producer's quote(safe="") does for this ASCII grammar, so the
    // canonical JSON is set raw: encoding it here would double-encode
    // ("{" would become %257B instead of %7B).
    if (windowLevel !== null) {
      params.set("window_level", canonicalWindowLevelJson(windowLevel));
    }
    if (visibleStructures !== null) {
      params.set("visible_structures", canonicalVisibleStructuresJson(visibleStructures));
    }
    return params;
  }

  // --- S3-008B §3.3 true W/L + structure visibility controls ---------------

  function updateWindowReadout() {
    const readout = document.getElementById("wl-readout");
    if (!readout) return;
    readout.textContent = windowLevel === null
      ? "auto"
      : "C=" + numberLex(windowLevel.center) + " W=" + numberLex(windowLevel.width);
  }

  function isStructureVisible(name) {
    // null resolves to ALL structures (the frozen default).
    return visibleStructures === null || visibleStructures.indexOf(name) !== -1;
  }

  function renderVisibilityGrid() {
    const holder = document.getElementById("visibility-grid");
    if (!holder) return;
    holder.replaceChildren();
    // Checkboxes are generated in FROZEN Structure order, never sorted.
    STRUCTURE_ORDER.forEach(function (name) {
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = isStructureVisible(name);
      checkbox.dataset.structure = name;
      checkbox.addEventListener("change", function () {
        applyVisibilityChange(name, checkbox.checked);
      });
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(name));
      holder.appendChild(label);
    });
  }

  function applyVisibilityChange(name, checked) {
    // Reset semantics (§3.3): unchecking reverts toward the ALL-visible
    // default; the canonical "all" spelling is null, not an explicit array.
    let next = visibleStructures === null
      ? STRUCTURE_ORDER.slice()
      : visibleStructures.slice();
    const at = next.indexOf(name);
    if (checked && at === -1) {
      next.push(name);
      next.sort(function (a, b) {
        return STRUCTURE_ORDER.indexOf(a) - STRUCTURE_ORDER.indexOf(b);
      });
    } else if (!checked && at !== -1) {
      next.splice(at, 1);
    }
    visibleStructures = next.length === STRUCTURE_ORDER.length ? null : next;
    renderVisibilityGrid();
    loadBundle(false).catch(function (error) { show(error.message, true); });
  }

  function applyWindowLevel() {
    const rawCenter = document.getElementById("wl-center").value.trim();
    const rawWidth = document.getElementById("wl-width").value.trim();
    if (rawCenter === "" || rawWidth === "") {
      show("Window / Level needs both a center and a width", true);
      return;
    }
    const center = Number(rawCenter);
    const width = Number(rawWidth);
    if (!Number.isFinite(center) || !Number.isFinite(width) || !(width > 1)) {
      show("Window width must be a finite number greater than 1", true);
      return;
    }
    windowLevel = {center: center, width: width};
    updateWindowReadout();
    loadBundle(false).catch(function (error) { show(error.message, true); });
  }

  function resetWindowLevel() {
    windowLevel = null;
    document.getElementById("wl-center").value = "";
    document.getElementById("wl-width").value = "";
    updateWindowReadout();
    loadBundle(false).catch(function (error) { show(error.message, true); });
  }

  function resetVisibility() {
    // §3.3: reset means visible_structures = null (ALL), not an ordering.
    visibleStructures = null;
    renderVisibilityGrid();
    loadBundle(false).catch(function (error) { show(error.message, true); });
  }

  function currentDraft() {
    // The first state request has no client-owned draft yet.  JSON null lets
    // the stateless server resolve the persisted applicable grid (or its
    // frozen all-included default); an empty array would be invalid.
    if (draftLineage.draft !== null) return cloneDraft(draftLineage.draft);
    if (!state || !state.inclusion_draft) return null;
    return cloneDraft(state.inclusion_draft);
  }

  function contextKey() {
    return queryView(false).toString();
  }

  function renderCached() {
    if (!bundleRows.length) return;
    const row = bundleRows.find(function (entry) { return entry.frame_index === frame; });
    const png = row && bundlePngs[String(row.frame_index)];
    if (!row || !png) return;
    createImageBitmap(new Blob([png], {type: "image/png"})).then(function (image) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0);
      image.close();
      document.getElementById("current-position").textContent =
        (slice + 1) + "/" + (state ? state.slices : "?") + " · " + (frame + 1) + "/" + (state ? state.frames : "?");
    }).catch(function () { show("Could not decode the frozen PNG frame", true); });
  }

  function decodeBundle(body) {
    const bytes = new Uint8Array(body);
    if (bytes.length < 8) throw new Error("short cine bundle");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const manifestLength = Number(view.getBigUint64(0, false));
    const decoder = new TextDecoder("utf-8", {fatal: true});
    const manifestBytes = bytes.slice(8, 8 + manifestLength);
    const manifest = JSON.parse(decoder.decode(manifestBytes));
    if (!manifest || manifest.version !== 1 || manifest.media !== "application/png-frame-bundle" || !Array.isArray(manifest.frames)) {
      throw new Error("invalid cine manifest");
    }
    const payload = bytes.slice(8 + manifestLength);
    bundleRows = manifest.frames;
    bundlePngs = Object.create(null);
    bundleRows.forEach(function (row) {
      if (!Number.isInteger(row.frame_index) || row.frame_index < 0 || !Number.isInteger(row.byte_offset) || !Number.isInteger(row.byte_length) || row.byte_offset < 0 || row.byte_length < 1 || row.byte_offset + row.byte_length > payload.length) {
        throw new Error("invalid cine frame row");
      }
      bundlePngs[String(row.frame_index)] = payload.slice(row.byte_offset, row.byte_offset + row.byte_length);
    });
    return manifest;
  }

  async function loadBundle(force) {
    if (!seriesUid) return;
    const nextContext = contextKey();
    if (!force && nextContext === bundleContext && bundle) { renderCached(); return; }
    const headers = {};
    if (bundle && bundle.etag) headers["If-None-Match"] = bundle.etag;
    const response = await fetch("/api/study/" + encodeURIComponent(seriesUid) + "/cine?" + nextContext, {headers: headers});
    if (response.status === 304) {
      bundleContext = nextContext;
      renderCached();
      return;
    }
    if (!response.ok) throw new Error("cine request failed (" + response.status + ")");
    const bytes = await response.arrayBuffer();
    const manifest = decodeBundle(bytes);
    bundle = {etag: response.headers.get("ETag"), manifest: manifest};
    bundleContext = nextContext;
    renderCached();
  }

  async function loadState() {
    if (!seriesUid) return;
    const response = await fetch("/api/study/" + encodeURIComponent(seriesUid) + "/state?" + queryView(true));
    if (!response.ok) throw new Error("state request failed (" + response.status + ")");
    const next = await response.json();
    const identityChanged = state && ["head_revision_id", "selection_id", "inclusion_set_id", "measurement_id", "qc_digest"].some(function (key) {
      return state[key] !== next[key];
    });
    state = next;
    // PR3 review r4310770076: the state projection carries the backend's own
    // DICOM-derived study identity (`cine.study_uid`). Capture it as the
    // independent bound-study identity the same-root no-op must match.
    noteBoundStudy(next.study_uid);
    // Round-2 P1: the bound study is only known once state resolves, so a
    // pending first-open intent is evaluated here as well as at startup.
    maybeConsumePendingOpen();
    draftLineage = updateDraftLineage(
      draftLineage,
      {type: "server_state", state: next}
    );
    syncDraftProjection();
    slice = next.slice;
    frame = next.frame;
    fps = next.fps;
    playing = next.playing;
    edFrame = next.selection ? next.selection.ed_frame : edFrame;
    esFrame = next.selection ? next.selection.es_frame : esFrame;
    renderGrid();
    renderVisibilityGrid();
    updateWindowReadout();
    // Durable lineage changes must revalidate even when the display context
    // string is unchanged; the response body remains the byte-level truth.
    if (identityChanged) await loadBundle(true);
    else if (contextKey() !== bundleContext) await loadBundle(false);
  }

  function renderGrid() {
    grid.replaceChildren();
    if (!state) return;
    (currentDraft() || []).forEach(function (row, index) {
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = row.included;
      checkbox.dataset.index = String(index);
      checkbox.addEventListener("change", function () {
        const editedDraft = currentDraft() || [];
        editedDraft[index].included = checkbox.checked;
        draftLineage = updateDraftLineage(
          draftLineage,
          {type: "edit", state: state, draft: editedDraft}
        );
        syncDraftProjection();
        loadBundle(false).catch(function (error) { show(error.message, true); });
      });
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode("slice " + row.slice_index + " " + row.frame_role + " " + row.structure));
      grid.appendChild(label);
    });
  }

  async function sendAction(action, params) {
    const local = updateDraftLineage(
      draftLineage,
      {type: "server_state", state: state}
    );
    const payload = {
      action: action,
      params: params || {},
      client_nonce: (crypto.randomUUID ? crypto.randomUUID() : String(Date.now())),
      view: {
        slice: slice,
        frame: frame,
        fps: fps,
        playing: playing,
        reference_sex: document.getElementById("reference-sex").value || null,
        inclusion_draft: local.draft
      },
      lineage: local.lineage
    };
    const response = await fetch("/api/study/" + encodeURIComponent(seriesUid) + "/action", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (response.status === 409) {
      state = result.state;
      draftLineage = updateDraftLineage(
        draftLineage,
        {type: "reset", state: result.state, draft: result.inclusion_draft}
      );
      syncDraftProjection();
      slice = state.slice;
      frame = state.frame;
      fps = state.fps;
      playing = state.playing;
      renderGrid();
      show("Stale view: review the reset draft and explicitly retry", true);
      await loadBundle(true);
      return;
    }
    if (!response.ok) throw new Error(result.error || "action failed");
    state = result.state;
    draftLineage = updateDraftLineage(
      draftLineage,
      {type: "commit", state: result.state, draft: result.inclusion_draft}
    );
    syncDraftProjection();
    renderGrid();
    show(action + " completed", false);
    await loadBundle(true);
  }

  function moveFrame(delta) {
    if (!state) return;
    frame = (frame + delta + state.frames) % state.frames;
    renderCached();
  }

  function moveSlice(delta) {
    if (!state) return;
    slice = (slice + delta + state.slices) % state.slices;
    loadBundle(false).catch(function (error) { show(error.message, true); });
  }

  function setPlaying(value) {
    playing = value;
    document.getElementById("play").textContent = playing ? "Pause" : "Play";
    loadBundle(false).catch(function (error) { show(error.message, true); });
    if (playing) playTick();
  }

  function playTick() {
    if (!playing) return;
    moveFrame(1);
    window.setTimeout(playTick, 1000 / fps);
  }

  async function chooseSeries(uid) {
    seriesUid = uid;
    state = null;
    draftLineage = {draft: null, lineage: null, dirty: false};
    bundle = null;
    bundleContext = "";
    try { await loadState(); show("Series loaded", false); } catch (error) {
      // No usable state answer: the bound-study identity stays unknown (never
      // a same-root match) until a later state answer establishes it.
      markBoundIdentityUnavailable();
      show(error.message, true);
    }
    updateReportLinks();
  }

  function updateReportLinks() {
    if (!seriesUid) return;
    const sex = document.getElementById("reference-sex").value;
    const suffix = sex ? "?reference_sex=" + encodeURIComponent(sex) : "";
    ["json", "csv", "pdf"].forEach(function (format) {
      document.getElementById("report-" + format).href =
        "/api/study/" + encodeURIComponent(seriesUid) + "/report." + format + suffix;
    });
  }

  async function loadStudies() {
    const response = await fetch("/api/studies");
    if (!response.ok) throw new Error("study list request failed");
    const studies = await response.json();
    const list = document.getElementById("series-list");
    studies.forEach(function (entry) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = entry.uid + " (" + entry.slices + "×" + entry.frames + ")";
      button.addEventListener("click", function () { chooseSeries(entry.uid); });
      item.appendChild(button);
      list.appendChild(item);
    });
    if (studies.length) await chooseSeries(studies[0].uid);
    // No series at all: there is no bound study this page could match, so the
    // identity is unknown rather than pending forever (PR3 review r4310770076).
    else markBoundIdentityUnavailable();
  }

  // --- Runtime-002 shell-owned switch affordance (profile b) ---------------
  // Target-selection ONLY: collects (source_root, stage_base, study_uid) and
  // invokes the ONE new Tauri IPC command `switch_study` by name with
  // plain-string args through the in-DOM `__TAURI_INTERNALS__.invoke` bridge
  // (re-injected on every navigation; no new dependency). No persistence, no
  // list, no clinical logic. Outside the desktop shell (plain browser, bridge
  // absent) the button explains that switching needs the desktop shell —
  // never a silent no-op. The shell owns the frozen order
  // (validate → terminate → spawn → handshake → verify → grant → navigate);
  // this page only requests and reports.
  let recoveryBanner = null;

  function dismissRecoveryBanner() {
    if (recoveryBanner && recoveryBanner.parentNode) {
      recoveryBanner.parentNode.removeChild(recoveryBanner);
    }
    recoveryBanner = null;
  }

  function showRecoveryBanner(text) {
    dismissRecoveryBanner();
    recoveryBanner = document.createElement("div");
    recoveryBanner.id = "recovery-banner";
    recoveryBanner.className = "recovery-banner error";
    recoveryBanner.setAttribute("role", "alert");

    const textSpan = document.createElement("span");
    textSpan.className = "recovery-banner-text";
    textSpan.textContent = text;
    recoveryBanner.appendChild(textSpan);

    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "recovery-banner-dismiss";
    closeBtn.textContent = "Dismiss";
    closeBtn.addEventListener("click", dismissRecoveryBanner);
    recoveryBanner.appendChild(closeBtn);

    const viewport = document.querySelector("section.viewport") || document.body;
    if (message && message.parentNode === viewport) {
      viewport.insertBefore(recoveryBanner, message);
    } else if (viewport.firstChild) {
      viewport.insertBefore(recoveryBanner, viewport.firstChild);
    } else {
      viewport.appendChild(recoveryBanner);
    }
  }

  function switchBridge() {
    if (window.__TAURI_INTERNALS__ && typeof window.__TAURI_INTERNALS__.invoke === "function") {
      return window.__TAURI_INTERNALS__.invoke.bind(window.__TAURI_INTERNALS__);
    }
    return null;
  }

  function mountSwitchAffordance() {
    if (typeof document.querySelector !== "function") return;
    const controls = document.querySelector("aside.controls");
    if (!controls) return;
    const section = document.createElement("section");
    const heading = document.createElement("h2");
    heading.textContent = "Switch study (desktop shell)";
    section.appendChild(heading);
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "Rebinds the owned backend to another staged study. Refusals keep the current study untouched.";
    section.appendChild(hint);
    const fields = [
      ["switch-source", "Source root (absolute path)"],
      ["switch-stage", "Stage base (absolute path)"],
      ["switch-study", "Study UID ([A-Za-z0-9._-])"]
    ];
    fields.forEach(function (pair) {
      const label = document.createElement("label");
      label.textContent = pair[1] + " ";
      const input = document.createElement("input");
      input.id = pair[0];
      input.type = "text";
      input.setAttribute("autocomplete", "off");
      input.setAttribute("spellcheck", "false");
      label.appendChild(input);
      section.appendChild(label);
    });
    const button = document.createElement("button");
    button.id = "switch-study";
    button.type = "button";
    button.textContent = "Switch Study";
    button.addEventListener("click", async function () {
      const invoke = switchBridge();
      if (!invoke) {
        show("Study switching needs the desktop shell (no Tauri bridge on this page).", true);
        return;
      }
      const sourceRootVal = document.getElementById("switch-source").value;
      const stageBaseVal = document.getElementById("switch-stage").value;
      const studyUidVal = document.getElementById("switch-study").value;
      const args = {
        source_root: sourceRootVal,
        stage_base: stageBaseVal,
        study_uid: studyUidVal,
        sourceRoot: sourceRootVal,
        stageBase: stageBaseVal,
        studyUid: studyUidVal
      };
      show("Switch requested: validating target before anything is terminated.", false);
      // First-open handoff (review P1-4): queue the opened identity BEFORE
      // invoking the shell. The shell kills backend A and navigates to the
      // bound authority before `switch_study` returns, so a post-invoke
      // continuation cannot POST this record to a live server. The cookie
      // (port-agnostic on the same loopback host) is consumed by the BOUND
      // DESTINATION page on load; every non-switched outcome and the
      // `#switch-failed` restored-page navigation clear it.
      //
      // Round-3 R3-1: exactly one handoff may be in flight, and THIS action
      // clears only the id it created. A page-global id would let an earlier
      // action's refusal delete a NEWER intent (probe: two clicks, first
      // refusal left the second cookie gone).
      if (!beginHandoff()) {
        show("A switch or reopen is already in progress; wait for it to finish.", true);
        return;
      }
      const entryPayload = {
        studyUid: args.study_uid,
        seriesUid: null,
        scanSourcePath: args.source_root,
        stageBaseDir: args.stage_base,
        destRoot: ""
      };
      const intentId = setPendingOpen(entryPayload);
      rememberActiveHandoff(intentId, entryPayload);
      // QUEUE-011: exactly ONE pending queue POST per click, AWAITED before
      // the shell invoke — the intent must be durable before backend A can
      // be killed. The handoff stays claimed across the await (single-flight
      // is released only in the invoke continuations below), so a second
      // click while the queue is in flight is still refused before any
      // cookie or invoke. Rejection-tolerant: a queue failure still invokes
      // (degraded, unrecorded switch — never a broken switch, never a
      // fabricated record).
      // Owner-ratified 2026-10-15: ANY queue error, timeout, or non-OK write
      // arms the explicit "may not be saved" notice (applied after outcome
      // handling below). Best-effort preserved: a queue failure still never
      // breaks the switch itself.
      let queueNoticeArmed = false;
      try {
        if (!await queuePendingOpen(intentId, Date.now() + 120000, entryPayload)) queueNoticeArmed = true;
      } catch (_) {
        queueNoticeArmed = true;
      }
      // F4: arm the opaque cross-navigation notice BEFORE the invoke — the
      // shell navigates away before this promise resolves, so the surviving
      // destination page (which never sees this page's continuation) reads
      // the notice from the cookie and renders the warning there.
      if (queueNoticeArmed) armRecentNotice("queue-error");
      invoke("switch_study", args).then(function (result) {
        // Binding-decision flow: a `needs_stage` result means the shell
        // validated the request but the staged destination does not exist
        // yet — NOTHING was terminated or spawned (A preserved). Stage on
        // the CURRENT owned backend via the EXISTING
        // `POST /api/library/stage` (filesystem-only, A untouched), then
        // re-invoke the IPC so the shell re-resolves (`reuse=true`) and
        // proceeds down the frozen switch order.
        let outcome = result;
        if (typeof outcome === "string") {
          try {
            outcome = JSON.parse(outcome);
          } catch (_) {
            // legacy plain text string fallback
          }
        }
        const isNeedsStage = (outcome && typeof outcome === "object" && outcome.status === "needs_stage") ||
                             (typeof outcome === "string" && outcome.indexOf("needs_stage:") === 0);
        if (isNeedsStage) {
          show("Target not staged yet: staging on the current backend, then retrying the switch.", false);
          let destPath = "";
          if (outcome && typeof outcome === "object" && outcome.dest_root) {
            destPath = outcome.dest_root;
          } else if (typeof outcome === "string") {
            // Unambiguous fallback: strip 'needs_stage:', detail is after the last colon (preserving Windows drive letters like C:)
            const rest = outcome.slice("needs_stage:".length);
            const lastColon = rest.lastIndexOf(":");
            destPath = (lastColon > 1) ? rest.slice(0, lastColon) : rest;
          }
          // The needs_stage sub-chain is RETURNED into the outer chain (not
          // detached) so the trailing timeout-notice step below also runs
          // after its terminal line.
          return fetch("/api/library/stage", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              source_path: args.source_root,
              study_uid: args.study_uid,
              dest_path: destPath
            })
          }).then(function (response) {
            if (!response.ok) throw new Error("stage request failed (" + response.status + ")");
            return response.json();
          }).then(function () {
            show("Staged: retrying the switch against the staged target.", false);
            // PR3 review 5564276070 F2: staging is unbounded (full source scan
            // + copy), so renew the CORRELATED intent before the final invoke —
            // same id, fresh exp, never overwriting a newer handoff.
            return renewPendingOpen(intentId, entryPayload).then(function (renewed) {
              if (!renewed) {
                queueNoticeArmed = true;
                armRecentNotice("queue-error");
              }
              return true; // continue after warning; invoke below
            });
          }).then(function () {
            return invoke("switch_study", args);
          }).then(function (retryResult) {
            if (retryResult && typeof retryResult === "object" && retryResult.status === "switched") {
              dismissRecoveryBanner();
              // C2 live-root tracking: remember the bound destination root
              // the shell actually granted, so a later reopen of THIS root
              // is a genuine same-root no-op (and a restaged same UID is
              // correctly NOT a no-op).
              if (typeof retryResult.dest_root === "string") noteLiveRoot(retryResult.dest_root);
              releaseActiveHandoff(intentId);
              endHandoff();
            } else {
              clearPendingOpen(intentId);
              releaseActiveHandoff(intentId);
              endHandoff();
            }
            const formatted = (retryResult && typeof retryResult === "object") ? JSON.stringify(retryResult) : retryResult;
            show("Switch result: " + formatted, false);
          }).catch(function (error) {
            clearPendingOpen(intentId);
            releaseActiveHandoff(intentId);
            endHandoff();
            show("Switch refused: " + error, true);
          });
        }
        if (outcome && typeof outcome === "object" && outcome.status === "switched") {
          dismissRecoveryBanner();
          if (typeof outcome.dest_root === "string") noteLiveRoot(outcome.dest_root);
          releaseActiveHandoff(intentId);
          endHandoff();
        } else {
          clearPendingOpen(intentId);
          releaseActiveHandoff(intentId);
          endHandoff();
        }
        const formatted = (outcome && typeof outcome === "object") ? JSON.stringify(outcome) : outcome;
        show("Switch result: " + formatted, false);
      }).catch(function (error) {
        clearPendingOpen(intentId);
        releaseActiveHandoff(intentId);
        endHandoff();
        show("Switch refused: " + error, true);
      }).then(function () {
        // R5: re-apply the queue notice AFTER outcome handling so the
        // terminal "Switch result/refused" line cannot silently erase the
        // owner-required "may not be saved" labeling.
        if (queueNoticeArmed) {
          queueNoticeArmed = false;
          show(((message.textContent) ? message.textContent + " " : "") + QUEUE_TIMEOUT_NOTICE, true);
        }
      });
    });
    section.appendChild(button);
    controls.appendChild(section);
  }
  mountSwitchAffordance();

  // --- Runtime-006 backend-owned recent cases persistence (inline) ----------
  let recentDigest = null;
  // Recent-list request sequencing (PR3 review r4310080007): every load takes
  // the next generation, and ONLY the newest generation may publish a digest
  // or replace the rendered list. Startup loading runs independently of
  // loadState, which can confirm a pending open, record it, and start a newer
  // refresh; without this guard a delayed startup response (old snapshot)
  // would erase the just-committed case and reset the digest to the stale
  // value, so the case looked unsaved until the next refresh.
  let recentListGeneration = 0;

  // Live-root tracking for the same-root no-op (review C2): the shell
  // reports the bound destination root on every successful switch
  // (`dest_root` on `switched` outcomes); the page remembers the latest
  // one as the live root. A reopen whose entry resolves to exactly that
  // root is a no-op; anything else (including the same UID under a new
  // stage base) goes through the shell.
  //
  // Harness seeding: `noteLiveRoot` is ALSO exposed on the IIFE's public
  // interface below (``__cmrTestHooks.noteLiveRoot``) so the executable
  // witnesses can replay the equivalent of a real granted outcome. The
  // page itself never calls the hook — it learns the root only from
  // genuine `switched` outcomes.
  let liveRootCanon = null;

  function canonicalRootString(value) {
    // Canonical comparison form for roots: forward slashes, no trailing
    // slash, drive-letter lowercased on Windows. Deliberately lexical —
    // the page cannot resolve links; containment/authority decisions stay
    // server-side. A non-string or blank root is not comparable (null).
    if (typeof value !== "string" || !value) return null;
    let norm = value.replace(/\\/g, "/").replace(/\/+$/, "");
    if (/^[A-Za-z]:/.test(norm)) norm = norm[0].toLowerCase() + norm.slice(1);
    return norm || null;
  }

  function currentLiveRoot() {
    return liveRootCanon;
  }

  function noteLiveRoot(root) {
    const canon = canonicalRootString(root);
    if (canon) liveRootCanon = canon;
    // PR3 review r4313043501: a new bound root is one of the two facts the
    // current-case marker is derived from, so re-derive it here as well. Without
    // this a row could keep advertising "current" (class, aria-current and the
    // visible label) after the live root moved to a different case root.
    reconcileRecentMarkers();
  }

  // --- Bound-root initialization (R7 review finding F5) --------------------
  // liveRootCanon is GENUINELY bound server state, not a client derivation:
  // the root this backend was spawned against is reported by the server it
  // was spawned for. The page fetches it on every initialization — before
  // and independent of any pending intent or recent-store health — so a
  // same-root Reopen after a plain reload/relaunch is an anchored no-op
  // instead of an unnecessary kill/respawn of the bound backend.
  //
  // Bounded initialization wait (R8 review finding 2, supplementary P2 fix):
  // The deadline race spans the COMPLETE fetch and body consumption operation
  // (`res.json()`). The abort controller remains active through body parsing,
  // the deadline timer is cleared only after full completion, and attempt
  // sequencing prevents a late-settling timed-out attempt from committing a
  // live root after a retry.
  var BOUND_ROOT_TIMEOUT_MS = (typeof window !== "undefined" && window.__CMR_BOUND_ROOT_TIMEOUT_MS) || 2500;
  let boundRootStatus = "idle"; // "idle" | "pending" | "settled" | "timed_out" | "failed"
  let boundRootReady = null;
  let boundRootAttempt = 0;

  async function loadBoundRoot() {
    boundRootStatus = "pending";
    const thisAttempt = ++boundRootAttempt;
    let timer = null;
    let controller = null;
    let timedOut = false;
    const timeoutMs = (typeof window !== "undefined" && window.__CMR_BOUND_ROOT_TIMEOUT_MS) || BOUND_ROOT_TIMEOUT_MS;
    try {
      if (typeof AbortController !== "undefined") {
        controller = new AbortController();
      }
      const signal = controller ? controller.signal : undefined;

      const timeoutPromise = new Promise(function (_, reject) {
        timer = setTimeout(function () {
          timedOut = true;
          if (controller) {
            try { controller.abort(); } catch (_) {}
          }
          const err = new Error("bound-root fetch timed out");
          err.code = "timeout";
          reject(err);
        }, timeoutMs);
      });

      const fetchAndParse = (async function () {
        try {
          const fetchOpts = signal ? { signal: signal } : {};
          const res = await fetch("/api/library/bound-root", fetchOpts);
          if (!res.ok) {
            return { ok: false, status: res.status };
          }
          const data = await res.json();
          return { ok: true, data: data };
        } catch (fetchErr) {
          return { ok: false, error: fetchErr };
        }
      })();

      const result = await Promise.race([fetchAndParse, timeoutPromise]);
      if (timer) clearTimeout(timer);

      // Isolation: if this attempt timed out or a newer attempt was started,
      // never commit live root or update status.
      if (timedOut || thisAttempt !== boundRootAttempt) {
        return null;
      }

      if (!result.ok) {
        boundRootStatus = "failed";
        return null;
      }
      const data = result.data;
      if (data && typeof data.root === "string" && data.root) {
        noteLiveRoot(data.root);
        boundRootStatus = "settled";
        // PR3 review r4311831218: the live bound root is one of the two facts
        // the current-case marker needs — reconcile once it is established.
        reconcileRecentMarkers();
        return data.root;
      }
      boundRootStatus = "settled";
      reconcileRecentMarkers();
      return null;
    } catch (err) {
      if (timer) clearTimeout(timer);
      if (controller) {
        try { controller.abort(); } catch (_) {}
      }
      if (thisAttempt === boundRootAttempt) {
        const isTimeout = timedOut || (err && (err.code === "timeout" || err.name === "AbortError"));
        boundRootStatus = isTimeout ? "timed_out" : "failed";
      }
      return null;
    }
  }

  // Review fix (delayed bound-root initialization): the recent list can
  // settle BEFORE the bound-root answer arrives, leaving a usable Reopen
  // button while currentLiveRoot() is still null — an early same-root click
  // would then decide from "unknown" and invoke the shell, killing and
  // respawning the live backend. Every decision that reads the live root
  // waits on THIS single initialization promise instead; it settles whether
  // the route answered (root noted) or failed (root stays unset — the no-op
  // remains off, the safe direction).
  function boundRootInitialized() {
    if (!boundRootReady) {
      boundRootReady = Promise.resolve()
        .then(function () { return loadBoundRoot(); })
        .catch(function () {
          boundRootStatus = "failed";
          return null;
        });
    }
    return boundRootReady;
  }

  // --- Bound study identity (PR3 review r4310770076) -----------------------
  // The same-root Reopen no-op may only assert "already open" when the live
  // backend's INDEPENDENTLY established bound study identity matches the
  // entry's studyUid. That identity is the backend's own DICOM-derived state
  // projection (`state.study_uid`, built from the cine volume it serves): the
  // recent row and the recent probe's echoed `study_uid` are client-supplied
  // echoes and prove nothing about the study the bound directory actually
  // contains. A stale row for study B whose directory now serves A must
  // therefore never answer "B is already open" while A is on screen — the
  // shell's prove step exists to reject that mismatch, and the page must not
  // short-circuit it. Unknown identity is never a match either: the decision
  // waits (bounded) for the first state answer and otherwise falls through to
  // the shell, which owns real mismatch rejection.
  var BOUND_IDENTITY_TIMEOUT_MS = (typeof window !== "undefined" && window.__CMR_BOUND_IDENTITY_TIMEOUT_MS) || 2500;
  let boundStudyUid = null;
  let boundIdentityStatus = "idle"; // "idle" | "pending" | "settled" | "timed_out" | "unavailable"
  let boundIdentityReady = null;
  let boundIdentityWaiters = [];
  let boundIdentityTimer = null;

  function settleBoundIdentity() {
    if (boundIdentityTimer) {
      clearTimeout(boundIdentityTimer);
      boundIdentityTimer = null;
    }
    const waiters = boundIdentityWaiters;
    boundIdentityWaiters = [];
    waiters.forEach(function (resolve) { resolve(boundStudyUid); });
  }

  function noteBoundStudy(value) {
    // Success: the backend reported the study it actually serves.
    if (typeof value !== "string" || !value) {
      markBoundIdentityUnavailable();
      return;
    }
    boundStudyUid = value;
    boundIdentityStatus = "settled";
    settleBoundIdentity();
    // PR3 review r4311831218: the bound study identity is the other fact the
    // current-case marker needs — a late first /state answer must reconcile the
    // already-rendered rows (the ordinary relaunch ordering).
    reconcileRecentMarkers();
  }

  function markBoundIdentityUnavailable() {
    // No usable state answer: identity stays UNKNOWN. A known identity is
    // never downgraded; the bounded wait still releases on its deadline.
    if (boundIdentityStatus === "settled") return;
    boundIdentityStatus = "unavailable";
    settleBoundIdentity();
  }

  function boundIdentityInitialized() {
    // Bounded wait for the first independently established identity. It
    // settles whether the state answer arrives (identity known) or the
    // deadline expires (unknown -> never "already open"). Callers read
    // `boundStudyUid` AFTER awaiting, so an identity that lands late is still
    // honoured.
    if (boundIdentityStatus === "settled" || boundIdentityStatus === "unavailable") {
      return Promise.resolve(boundStudyUid);
    }
    if (!boundIdentityReady) {
      boundIdentityStatus = "pending";
      boundIdentityReady = new Promise(function (resolve) {
        boundIdentityWaiters.push(resolve);
        boundIdentityTimer = setTimeout(function () {
          if (boundIdentityStatus !== "settled") boundIdentityStatus = "timed_out";
          settleBoundIdentity();
        }, BOUND_IDENTITY_TIMEOUT_MS);
      });
    }
    return boundIdentityReady;
  }

  // --- Server-resolved destination probe (shared) --------------------------
  // The page never decides workspace identity from a client lexical
  // derivation. `dest_root` is the SERVER's own resolved derivation for an
  // entry: the backend applies resolve_path(stageBaseDir) + studyUid and then
  // physically resolves the result, so anchored equivalents such as
  // `/stage/./` answer the same root as `/stage`. PR3 review #4 (app.js:1230):
  // pending-open confirmation compares the bound root against THIS value too —
  // the pending slot preserves the raw stage spelling, so a lexical client
  // derivation mis-compares it and silently strands a legitimate open.
  //
  // Bounded discipline (unchanged from the Reopen probe, reused verbatim):
  // the deadline spans the COMPLETE fetch and body consumption (`res.json()`),
  // the abort controller stays active through body parsing, and `isStale()`
  // is evaluated after settlement so a superseded or timed-out attempt can
  // never publish a root.
  function probeTimeoutMs() {
    var configured = (typeof window !== "undefined" && window.__CMR_PROBE_TIMEOUT_MS) || 0;
    return configured > 0 ? configured : 2500;
  }

  // --- Bounded write acknowledgement / cleanup (PR3 review 5564032012 F2) ---
  // The record POST is the only write whose outcome decides whether this page
  // may release the pending handoff, so its deadline must cover the WHOLE
  // exchange (POST headers AND body, the 409 stale-digest body plus the retry
  // POST, and the ack body) in one coherent budget. Ancillary work — the
  // post-save list refresh and the pending-slot clear — is bounded separately
  // and never decides the commit outcome.
  var RECORD_ACK_TIMEOUT_MS = 8000;
  var PENDING_CLEANUP_TIMEOUT_MS = 1500;

  function recordAckTimeoutMs() {
    var configured = (typeof window !== "undefined" && window.__CMR_RECORD_ACK_TIMEOUT_MS) || 0;
    return configured > 0 ? configured : RECORD_ACK_TIMEOUT_MS;
  }

  function pendingCleanupTimeoutMs() {
    var configured = (typeof window !== "undefined" && window.__CMR_PENDING_CLEANUP_TIMEOUT_MS) || 0;
    return configured > 0 ? configured : PENDING_CLEANUP_TIMEOUT_MS;
  }

  function fetchResolvedDestRoot(entry, isStale) {
    // Resolves {status, data, destRoot}: status is "settled" | "failed" |
    // "timed_out" | "stale"; destRoot is the server-resolved destination
    // (null when the answer carried none); data is the probe envelope.
    const timeoutMs = probeTimeoutMs();
    let timer = null;
    let controller = null;
    let timedOut = false;
    if (typeof AbortController !== "undefined") {
      controller = new AbortController();
    }
    const signal = controller ? controller.signal : undefined;
    const timeoutPromise = new Promise(function (_, reject) {
      timer = setTimeout(function () {
        timedOut = true;
        if (controller) {
          try { controller.abort(); } catch (_) {}
        }
        const err = new Error("entry probe timed out");
        err.code = "timeout";
        reject(err);
      }, timeoutMs);
    });
    const probeParams = new URLSearchParams({
      source: entry.scanSourcePath,
      stage: entry.stageBaseDir,
      study: entry.studyUid
    });
    const fetchAndParse = (async function () {
      try {
        const fetchOpts = signal ? { signal: signal } : {};
        const res = await fetch("/api/library/recent/probe?" + probeParams.toString(), fetchOpts);
        if (!res.ok) {
          return { ok: false, status: res.status };
        }
        const data = await res.json();
        return { ok: true, data: data };
      } catch (fetchErr) {
        return { ok: false, error: fetchErr };
      }
    })();
    return Promise.race([fetchAndParse, timeoutPromise])
      .then(function (result) {
        if (timer) clearTimeout(timer);
        if (timedOut || isStale()) return { status: "stale", data: null, destRoot: null };
        if (!result || !result.ok || !result.data) {
          return { status: "failed", data: null, destRoot: null };
        }
        const dest = result.data.dest_root;
        return {
          status: "settled",
          data: result.data,
          destRoot: (typeof dest === "string" && dest) ? dest : null
        };
      })
      .catch(function (err) {
        if (timer) { try { clearTimeout(timer); } catch (_) {} }
        if (controller) {
          try { controller.abort(); } catch (_) {}
        }
        const isTimeout = timedOut || (err && (err.code === "timeout" || err.name === "AbortError"));
        return { status: isTimeout ? "timed_out" : "failed", data: null, destRoot: null };
      });
  }

  // --- Cross-navigation queue-failure notice (R7 review finding F4) --------
  // A queue failure armed before the shell invoke must survive the
  // navigation the shell performs BEFORE the invoke resolves: the outgoing
  // page's continuation never runs, so the warning cannot live only in its
  // message log. It is carried in an opaque, path-free notice cookie that
  // the surviving destination consumes exactly once at startup. The payload
  // carries no entry, no filesystem path, and no free text — a version, an
  // expiry, and a fixed status code.
  const RECENT_NOTICE_COOKIE = "cmr_recent_notice";
  const RECENT_NOTICE_MESSAGES = {
    "queue-error": "Recent case may not be saved: the pending queue write did not complete, so the last switch proceeded unrecorded."
  };

  function armRecentNotice(code) {
    if (!Object.prototype.hasOwnProperty.call(RECENT_NOTICE_MESSAGES, code)) return;
    try {
      const payload = {v: 1, exp: Date.now() + 120000, code: code};
      document.cookie = RECENT_NOTICE_COOKIE + "=" + encodeURIComponent(JSON.stringify(payload)) +
        "; path=/; max-age=120; SameSite=Lax";
    } catch (_) {
      // Best-effort: a cookie failure must never break the switch itself.
    }
  }

  function readRecentNotice() {
    try {
      let raw = null;
      document.cookie.split(";").forEach(function (pair) {
        const idx = pair.indexOf("=");
        if (idx > 0 && pair.slice(0, idx).trim() === RECENT_NOTICE_COOKIE) {
          raw = pair.slice(idx + 1);
        }
      });
      if (!raw) return null;
      const parsed = JSON.parse(decodeURIComponent(raw));
      if (!parsed || typeof parsed !== "object") return null;
      if (parsed.v !== 1) return null;
      if (typeof parsed.code !== "string" ||
          !Object.prototype.hasOwnProperty.call(RECENT_NOTICE_MESSAGES, parsed.code)) return null;
      if (typeof parsed.exp !== "number" || Date.now() > parsed.exp) return null;
      return parsed;
    } catch (_) {
      return null;
    }
  }

  function clearRecentNotice() {
    try {
      document.cookie = RECENT_NOTICE_COOKIE + "=; path=/; max-age=0; SameSite=Lax";
    } catch (_) {}
  }

  // Eligibility snapshot at script evaluation, mirroring `incomingIntent`: the
  // notice armed by the page we navigated FROM is captured before ANY click
  // handler on THIS page can arm a notice, so a page can never consume its own
  // outgoing warning.
  //
  // PR3 review r4310770084: the cookie is NOT cleared here any more. Clearing
  // at snapshot time retired the warning before it was ever displayed, so a
  // page that could not deliver it (stalled recent-list load) and any reload
  // lost it permanently. It is retired by `consumeRecentNotice()` AFTER the
  // warning is shown.
  const incomingNotice = readRecentNotice();

  function consumeRecentNotice() {
    // Exactly once per page load, from the eval-time snapshot: a stale or
    // unknown notice renders nothing.
    if (!incomingNotice) return;
    const text = RECENT_NOTICE_MESSAGES[incomingNotice.code];
    show(text, true);
    try {
      const banner = document.getElementById("recent-banner");
      if (banner && banner.style.display !== "block") showRecentBanner(text);
    } catch (_) {}
    // Retire only AFTER delivery (review r4310770084): the destination of a
    // failed queue write must not lose the warning to a load that never
    // settles, and a reload before display still finds it.
    clearRecentNotice();
  }

  // --- First-open handoff (review P1-4, round-2 P1 binding) ----------------
  // `switch_study` kills the current backend and navigates to the bound
  // authority BEFORE the IPC call returns (desktop main.rs: kill A -> spawn
  // B -> navigate -> return), so neither a first open nor a reopen record
  // can be POSTed from the outgoing page: its origin's server is already
  // dead. The intent is queued on a SAME-HOST cookie BEFORE the invoke
  // (cookies are port-agnostic, so the new origin can read it).
  //
  // Round-2 P1: a cookie alone proves nothing about WHO consumes it. The
  // previous consumer read "whichever cookie exists after an async startup
  // GET", so an outgoing page with a still-pending startup request consumed
  // its OWN freshly queued intent and recorded an unswitched target
  // (VERIFIED_BY_EXECUTION). Four guards make consumption specific:
  //   1. eligibility is snapshotted SYNCHRONOUSLY at page init, before any
  //      switch can be requested on this page (`incomingIntent`);
  //   2. the producer stamps its own authority, so neither it nor a reload of
  //      it can consume its own intent (`producer === location.host`);
  //   3. binding: the consumer's currently bound study (`state.study_uid`,
  //      always present in the state projection) must equal the intent's
  //      target study, otherwise the cookie is dropped unrecorded;
  //   4. every clear/consume is correlated by `id`, and consumption happens
  //      at most once per page (`consumedIntentId`).
  // `exp` is a backstop for an app closed mid-switch; `#switch-failed`
  // restored pages discard the intent before any consumption runs.
  const PENDING_OPEN_COOKIE = "cmr_pending_open";
  let incomingIntent = null;    // eligibility snapshot taken at page init
  let consumedIntentId = null;  // consume at most once per page
  let handoffOutstanding = false; // round-3 R3-1: single-flight handoff
  // PR3 review F3 (retained handoff hint): the pending-open confirmation is
  // single-flight, bounded, and deadline-guarded. The correlated cookie and
  // the durable pending slot are released ONLY on a confirmed record or a
  // deliberate refusal/expiry — a transient read failure keeps them for a
  // later retry (or reload) instead of stranding a successful open.
  var PENDING_CONFIRM_MAX_ATTEMPTS = 3;
  var PENDING_CONFIRM_RETRY_DELAY_MS = 400;
  var PENDING_CONFIRM_TIMEOUT_MS = 2500;
  let pendingConfirmInFlight = null;    // single-flight {id}
  let pendingConfirmTimer = null;       // bounded retry handle
  let pendingConfirmAttempts = 0;       // attempts for the captured hint id
  let pendingConfirmExhaustedId = null; // terminal for this page; a reload retries
  let pendingConfirmSeq = 0;            // late-settlement isolation (slot read)
  let pendingDestinationProbe = 0;      // late-settlement isolation (server resolve)
  let incomingHandoffClaim = null;      // incoming intent reserved through confirmation
  let incomingClaimExpiryTimer = null;  // claim-armed expiry cleanup (review 5566059819 P2-2)
  let activeOutgoingHandoff = null;     // active producer lease (id+entry) survives cookie Max-Age

  function readPendingIntent() {
    // Pure reader: returns the parsed, unexpired hint or null. The hint
    // carries NO entry/paths (confidentiality review P1) — only the opaque
    // slot id, expiry, and producer stamp.
    try {
      let raw = null;
      document.cookie.split(";").forEach(function (pair) {
        const idx = pair.indexOf("=");
        if (idx > 0 && pair.slice(0, idx).trim() === PENDING_OPEN_COOKIE) {
          raw = pair.slice(idx + 1);
        }
      });
      if (!raw) return null;
      const parsed = JSON.parse(decodeURIComponent(raw));
      if (!parsed || typeof parsed !== "object") return null;
      if (typeof parsed.id !== "string" || !parsed.id) return null;
      if (typeof parsed.producer !== "string" || !parsed.producer) return null;
      if (typeof parsed.exp !== "number" || Date.now() > parsed.exp) return null;
      return parsed;
    } catch (_) {
      return null;
    }
  }

  function writePendingHint(id, exp) {
    // Correlated hint writer: the SAME id can be re-written with a fresh exp
    // (staging renewal, PR3 review 5564276070 F2) without minting a new id, so
    // the durable open id — and its server-side replay idempotence — survive.
    const hint = {id: id, exp: exp, producer: window.location.host};
    try {
      document.cookie = PENDING_OPEN_COOKIE + "=" + encodeURIComponent(JSON.stringify(hint)) +
        "; path=/; max-age=" + Math.max(1, Math.ceil((exp - Date.now()) / 1000)) + "; SameSite=Lax";
    } catch (_) {
      // Best-effort: a cookie failure must never break the switch itself.
    }
  }

  function setPendingOpen(entry) {
    // Confidentiality (review P1): the cookie is port-agnostic on the loopback
    // host, so it is JS-readable FAR beyond the producing origin. It therefore
    // carries ONLY the opaque handoff hint {id, exp, producer} — NEVER the
    // entry (study paths must not transit JS-readable storage). Cookie write
    // only: the server-side queue is the caller's single awaited
    // queuePendingOpen POST (QUEUE-011) — this function fires no POST itself,
    // so each click queues exactly once. The bound destination consumes the
    // hint (confirmPendingOpen) and clears the slot. A queue failure keeps
    // the old cookie behavior's best-effort contract: the switch itself must
    // never break for it — the destination then finds no intent and simply
    // does not record.
    const id = "po" + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
    writePendingHint(id, Date.now() + 120000);
    return id;
  }

  async function queuePendingOpen(id, exp, entry) {
    // Server-side pending slot: same-origin POST (auth by construction),
    // same frozen envelope, no-store.
    //
    // QUEUE-011: exactly ONE queue POST per click, AWAITED by the caller
    // before the shell invoke — the intent must be durable before backend
    // A can be killed. Resolves true when the slot write is acknowledged;
    // rejects on transport failure (callers treat failure as best-effort:
    // the switch still invokes, degrading to an unrecorded switch rather
    // than a broken one — never a fabricated record).
    //
    // R5 bounded queue timeout (QUEUE-HANG): a queue POST that HANGS
    // (backend A killed mid-flight, connection never settles nor resets)
    // must degrade EXACTLY like a queue failure — the await settles within
    // QUEUE_TIMEOUT_MS and the caller still invokes (unrecorded switch,
    // never broken, never fabricated). AbortController cancels the real
    // fetch when available; the timeout race guarantees the await settles
    // even on runtimes without AbortController or when the fetch ignores
    // the abort signal.
    var QUEUE_TIMEOUT_MS = (typeof window !== "undefined" && window.__CMR_QUEUE_TIMEOUT_MS) || 5000;
    const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    let timer = null;
    try {
      const timeout = new Promise(function (_, reject) {
        timer = setTimeout(function () {
          if (controller) {
            try { controller.abort(); } catch (_) {}
          }
          const timeoutErr = new Error("queuePendingOpen timeout");
          timeoutErr.code = "queue-timeout";
          reject(timeoutErr);
        }, QUEUE_TIMEOUT_MS);
      });
      const fetchOpts = {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({id: id, exp: exp, entry: entry})
      };
      if (controller) {
        fetchOpts.signal = controller.signal;
      }
      const attempt = fetch("/api/library/recent/pending", fetchOpts);
      // Swallow the late settlement of the hung fetch so a post-timeout
      // resolve/reject never surfaces as an unhandled rejection; the race
      // below already decided the outcome.
      attempt.then(null, function () {});
      const res = await Promise.race([attempt, timeout]);
      return !!(res && res.ok);
    } finally {
      if (timer !== null) { try { clearTimeout(timer); } catch (_) {} }
    }
  }

  function readRawPendingHint() {
    // Unfiltered correlated-hint reader: unlike readPendingIntent this does NOT
    // drop an expired hint, because a still-staging producer must be able to
    // recognize ITS OWN (possibly expired) id and renew it. Eligibility at page
    // init still uses readPendingIntent, so abandoned intents keep expiring.
    try {
      let raw = null;
      document.cookie.split(";").forEach(function (pair) {
        const idx = pair.indexOf("=");
        if (idx > 0 && pair.slice(0, idx).trim() === PENDING_OPEN_COOKIE) {
          raw = pair.slice(idx + 1);
        }
      });
      if (!raw) return null;
      const parsed = JSON.parse(decodeURIComponent(raw));
      if (!parsed || typeof parsed !== "object") return null;
      if (typeof parsed.id !== "string" || !parsed.id) return null;
      return parsed;
    } catch (_) {
      return null;
    }
  }

  function rememberActiveHandoff(id, entry) {
    // PR3 review 5564803264 F1: keep correlated ownership in the producer page's
    // live JS state while shell staging runs. Browser cookie Max-Age removes the
    // cookie from document.cookie after 120s, so a raw-cookie reader cannot renew
    // an actually expired cookie. This lease is not persisted: abandoning/closing
    // the producer destroys it, leaving the cookie+slot's original exp to bound
    // the abandoned intent. The full entry remains in memory only (never in the
    // cookie).
    activeOutgoingHandoff = {id: id, entry: entry, exp: Date.now() + 120000};
  }

  function releaseActiveHandoff(id) {
    if (activeOutgoingHandoff && activeOutgoingHandoff.id === id) {
      activeOutgoingHandoff = null;
    }
  }

  async function renewPendingOpen(intentId, entry) {
    // PR3 review 5564803264 F1: during a live staging chain the browser may
    // physically remove the cookie at positive Max-Age; JavaScript cannot read
    // or revive that deleted value. The producer therefore keeps a small,
    // in-memory active lease (id + entry) from before its first invoke. Only
    // this still-running producer may renew without the cookie, and only while
    // no different/newer cookie is present. The cookie itself remains
    // path-free; closing/abandoning the page destroys the lease and the original
    // cookie + slot TTL still bounds abandoned handoffs.
    const current = readRawPendingHint();
    const lease = activeOutgoingHandoff;
    if (!lease || lease.id !== intentId || lease.entry !== entry) return false;
    if (current && current.id !== intentId) return false;
    const exp = Date.now() + 120000;
    lease.exp = exp;
    writePendingHint(intentId, exp);
    try {
      return await queuePendingOpen(intentId, exp, entry);
    } catch (_) {
      return false;
    }
  }

  // R5 owner-required notice (shared by both legs; RATIFIED 2026-10-15 for
  // ANY queue error or timeout): armed by the queue catch/non-OK below and
  // applied in each leg's trailing continuation AFTER outcome handling, so
  // the terminal result/refused line keeps its outcome AND the "may not be
  // saved" labeling survives. Honest labeling, not a block: the switch
  // still proceeds (degraded unrecorded switch, never broken, never
  // fabricated).
  var QUEUE_TIMEOUT_NOTICE = "Recent case may not be saved: the pending queue timed out, so this switch proceeds unrecorded.";

  function pendingConfirmTimeoutMs() {
    var configured = (typeof window !== "undefined" && window.__CMR_PENDING_TIMEOUT_MS) || 0;
    return configured > 0 ? configured : PENDING_CONFIRM_TIMEOUT_MS;
  }

  function pendingConfirmMaxAttempts() {
    var configured = (typeof window !== "undefined" && window.__CMR_PENDING_MAX_ATTEMPTS) || 0;
    return configured > 0 ? configured : PENDING_CONFIRM_MAX_ATTEMPTS;
  }

  function pendingConfirmRetryDelayMs() {
    var configured = (typeof window !== "undefined" && window.__CMR_PENDING_RETRY_DELAY_MS) || 0;
    return configured > 0 ? configured : PENDING_CONFIRM_RETRY_DELAY_MS;
  }

  async function readPendingSlotOnce() {
    // PR3 review F3 (full fetch+body deadline): the deadline covers the HEADERS
    // and the BODY, and a late settlement of an abandoned attempt is isolated
    // by sequence, so a timed-out read can never commit a stale confirmation.
    const thisAttempt = ++pendingConfirmSeq;
    let timer = null;
    let controller = null;
    let timedOut = false;
    try {
      if (typeof AbortController !== "undefined") {
        controller = new AbortController();
      }
      const signal = controller ? controller.signal : undefined;
      const timeoutPromise = new Promise(function (_, reject) {
        timer = setTimeout(function () {
          timedOut = true;
          if (controller) { try { controller.abort(); } catch (_) {} }
          const err = new Error("pending open read timed out");
          err.code = "timeout";
          reject(err);
        }, pendingConfirmTimeoutMs());
      });
      const fetchAndParse = (async function () {
        const fetchOpts = signal ? {signal: signal} : {};
        const res = await fetch("/api/library/recent/pending", fetchOpts);
        if (!res.ok) {
          const httpErr = new Error("pending open read failed (" + res.status + ")");
          httpErr.code = "http";
          throw httpErr;
        }
        return await res.json();
      }());
      // A late settlement of an abandoned attempt must never surface as an
      // unhandled rejection; the race below already decided the outcome.
      fetchAndParse.then(null, function () {});
      const data = await Promise.race([fetchAndParse, timeoutPromise]);
      if (timedOut || thisAttempt !== pendingConfirmSeq) return {ok: false, stale: true};
      return {ok: true, data: data};
    } catch (error) {
      return {ok: false, error: error};
    } finally {
      if (timer !== null) { try { clearTimeout(timer); } catch (_) {} }
    }
  }

  async function confirmPendingOpen(hint) {
    // Bound destination only: fetch the queued intent, verify it names THIS
    // hint id and the bound study, record the successfully opened identity
    // through the normal POST path (server-authoritative counters,
    // stale-digest merge), then clear the slot.
    //
    // PR3 review F3 outcome contract: "confirmed" (record persisted, slot
    // cleared), "refused" (deliberate: the slot is absent or id-mismatched —
    // the correlated hint is released), "standby" (this page is not the
    // destination: a different study UID, or a different/unknown workspace
    // root, or an unresolvable server destination — hint and durable slot are
    // LEFT INTACT for the real destination), "failed" (the record did NOT
    // commit: hint + slot retained, never reported as confirmed), "retry"
    // (read-side or server-resolve failure: transport, non-OK, deadline,
    // unparseable body, missing resolved destination — the caller retries
    // with the SAME idempotent open id).
    const read = await readPendingSlotOnce();
    if (!read.ok) return "retry";
    const pending = read.data;
    if (!pending || pending.pending !== true || !pending.entry) return "refused";
    if (pending.id !== hint.id) return "refused";
    // PR3 review #4 (app.js:1219, different-UID consumer): a different study
    // UID only proves that THIS page is not the destination. The hint cookie is
    // shared across loopback ports and the pending slot uses the same per-user
    // store, so retiring either here strands the real destination: it would
    // find no hint at its own load and the durable slot would never be
    // consumed. Stand by — zero records, zero clears, handoff untouched — and
    // let the correlated hint/slot expire naturally if no destination comes.
    // PR3 review r4311831210 (app.js:1501): the page state is transiently null
    // while a NORMAL series change (chooseSeries) awaits its next /state answer,
    // and a read that completes inside that window proves NOTHING about the
    // destination. Reading null as "different study" returned "standby", which
    // the caller records as TERMINAL (pendingConfirmExhaustedId): no later state
    // answer ever retried, so a successfully opened case stayed unrecorded with
    // no notice at all. Unknown is never proof: RETRY the bounded budget with the
    // correlated hint and the durable slot retained, so the decision is made on
    // the identity that actually lands; only a KNOWN, DIFFERENT study UID stands
    // by. Exhaustion ends in the actionable "may not be saved" notice rather than
    // a silent no-op — the same fail-closed-by-retrying rule the bound-root
    // lookup above already follows.
    if (!state) return "retry";
    if (typeof state.study_uid !== "string" || !state.study_uid) return "retry";
    if (state.study_uid !== pending.entry.studyUid) return "standby";
    // PR3 review #3 (root binding): the study UID alone is NOT a destination
    // identity — the same UID can be staged under different roots, and the
    // hint cookie plus the pending slot are shared across loopback ports. A
    // same-UID window on another root must never record the queued target,
    // clear the shared slot/hint, and strand the real handoff. Await the
    // SERVER-REPORTED bound root for THIS backend and compare it with the
    // pending entry's SERVER-RESOLVED destination. Only a match may confirm.
    //
    // PR3 review #4 (app.js:1230): the pending slot preserves the raw validated
    // `stageBaseDir` spelling (e.g. `/stage/./`), while the bound root is
    // physically resolved; a client lexical derivation therefore mis-compares
    // anchored equivalents and would permanently stand by on a legitimate
    // destination, leaving the successful open silently unrecorded. Resolve the
    // destination on the SERVER (the same probe Reopen uses), never from the
    // display-only `destRoot`, and compare only server-resolved values.
    await boundRootInitialized();
    // PR3 review #5 (app.js:1316): a FAILED or timed-out bound-root lookup is
    // NOT proof that this page is a different workspace. Reading it as one here
    // (standby) silently set pendingConfirmExhaustedId, so neither the heartbeat
    // nor the pending retry budget ever tried again and a genuinely successful
    // switch stayed unrecorded until the retained hint expired, with no
    // indication. Binding stays fail-closed by RETRYING instead: drop the
    // memoized failed initialization (so the next attempt genuinely re-asks
    // /api/library/bound-root) and let the existing bounded budget decide -
    // exhaustion ends in the actionable "may not be saved" notice with the hint
    // and the durable slot retained.
    if (boundRootStatus === "timed_out" || boundRootStatus === "failed") {
      boundRootReady = null;
      return "retry";
    }
    const boundRoot = canonicalRootString(currentLiveRoot());
    // Settled with no root: the server itself reports no bound workspace root,
    // so this backend is not the destination (fail-closed binding preserved).
    if (!boundRoot) return "standby";
    const thisProbe = ++pendingDestinationProbe;
    const probe = await fetchResolvedDestRoot(pending.entry, function () {
      return thisProbe !== pendingDestinationProbe;
    });
    // A superseded probe, a failed/timed-out probe, or an answer without a
    // resolved destination never decides anything: retry under the same
    // bounded budget (hint + slot retained; never a silent mismatch).
    if (probe.status === "stale") return "retry";
    if (probe.status !== "settled" || !probe.destRoot) return "retry";
    const entryDest = canonicalRootString(probe.destRoot);
    if (!entryDest || entryDest !== boundRoot) return "standby";
    try {
      await recordRecentCase(pending.entry, hint.id);
    } catch (error) {
      if (error && error.code === "record-unknown") {
        // PR3 review 5564032012 F2: the write may or may not have committed
        // (deadline / transport). Keep the hint + durable slot and retry with
        // the SAME idempotent open id, so a committed write is never lost and a
        // replay is never counted twice.
        return "retry";
      }
      show("Recent case record failed: " + error, true);
      return "failed";
    }
    // F5: the live root is NOT seeded from this entry — the pending entry is
    // client-queued data (its stage base is server-validated, but it is not
    // the granted root). The page learns its genuinely bound root from the
    // server's bound-root route (loadBoundRoot), on every initialization,
    // independent of this consumption path; a null/blank answer leaves the
    // live root unset (no-op stays off, the safe direction).
    //
    // PR3 review 5564032012 F2: the cleanup acknowledgement is bounded — a hung
    // clear can never block or downgrade this KNOWN commit (the durable slot
    // expires by its own `exp`, and a later replay is deduplicated by the
    // persistent open id).
    await clearPendingSlotBounded(hint.id);
    return "confirmed";
  }

  function schedulePendingConfirmRetry(live) {
    pendingConfirmTimer = setTimeout(function () {
      pendingConfirmTimer = null;
      const again = readPendingIntent();
      if (again && again.id === live.id) startPendingConfirmation(live);
    }, pendingConfirmRetryDelayMs());
  }

  function failPendingConfirmation(live) {
    // Bounded retry exhausted (or an unexpected throw): keep the correlated
    // hint and the durable slot, and surface an actionable failure — never a
    // silent no-op. A later page load retries with the same idempotent id.
    pendingConfirmExhaustedId = live.id;
    show("Recent case may not be saved: the pending case could not be confirmed; " +
      "reload the page to retry.", true);
  }

  function claimIncomingHandoff(live) {
    // PR3 review 5564276070 F1: an eligible INCOMING destination intent must own
    // the shared handoff slot until its record outcome is KNOWN (ack, deliberate
    // refusal, or natural expiry). Without this reservation a Switch/Reopen to
    // another case clicked while B confirms is permitted, `setPendingOpen`
    // replaces the shared hint and `queuePendingOpen` replaces the only durable
    // slot; if that second switch is then refused, B is still open but its
    // original successful open can no longer be recorded (its id is lost even on
    // reload). `beginHandoff` is the existing single-flight guard, so this adds
    // no second mechanism.
    if (incomingHandoffClaim === live.id) return true;
    if (!beginHandoff()) return false;
    incomingHandoffClaim = live.id;
    armIncomingClaimExpiry(live);
    return true;
  }

  function clearIncomingClaimExpiry() {
    if (incomingClaimExpiryTimer === null) return;
    try { clearTimeout(incomingClaimExpiryTimer); } catch (_) {}
    incomingClaimExpiryTimer = null;
  }

  function armIncomingClaimExpiry(live) {
    // PR3 review 5566059819 P2-2: the reservation is armed for cleanup at the
    // moment it is CLAIMED, not only once a confirmation starts. A hung or
    // failed /studies or /state (confirmation never runs) can therefore no
    // longer hold this page's single handoff slot past the intent's own exp.
    // The callback releases ONLY this page's correlated claim: an already
    // released claim (a newer action now owns the slot) is a no-op, and the
    // shared hint/slot of another window is never written or cleared.
    clearIncomingClaimExpiry();
    const exp = (live && typeof live.exp === "number") ? live.exp : 0;
    const delay = Math.max(0, exp - Date.now()) + 100;
    incomingClaimExpiryTimer = setTimeout(function () {
      incomingClaimExpiryTimer = null;
      if (incomingHandoffClaim !== live.id) return;
      incomingHandoffClaim = null;
      if (incomingIntent && incomingIntent.id === live.id) incomingIntent = null;
      endHandoff();
    }, delay);
  }

  function releaseIncomingHandoff(live) {
    if (incomingHandoffClaim !== live.id) return;
    incomingHandoffClaim = null;
    clearIncomingClaimExpiry();
    endHandoff();
  }

  function releaseStaleIncomingClaim() {
    // PR3 review 5566059819 P2-2: a handoff attempt must not be blocked by a
    // reservation whose hint is already gone, expired, or replaced. Release is
    // strictly correlated: only the claim this page still holds is dropped
    // (never another action's active handoff), and while the claimed hint is
    // live the reservation stays because its confirmation may still record it.
    if (incomingHandoffClaim === null) return;
    const live = readPendingIntent();
    if (live && live.id === incomingHandoffClaim) return;
    const staleId = incomingHandoffClaim;
    incomingHandoffClaim = null;
    clearIncomingClaimExpiry();
    if (incomingIntent && incomingIntent.id === staleId) incomingIntent = null;
    endHandoff();
  }

  function holdIncomingHandoffUntilExpiry(live) {
    // Unknown outcome after the retry budget is exhausted: keep the reservation
    // until the intent's OWN expiry so a new handoff cannot erase the retained
    // id, then release the page controls. The pre-existing 120s TTL still bounds
    // the hold — nothing is extended globally. Reuses the claim-armed cleanup so
    // there is exactly ONE correlated release path.
    armIncomingClaimExpiry(live);
  }

  function startPendingConfirmation(live) {
    // Single-flight: at most one read attempt at a time per page, so
    // concurrent callers (startup + state heartbeat) never double-read or
    // double-record; the bounded retry itself re-enters here.
    if (!claimIncomingHandoff(live)) return;
    pendingConfirmInFlight = {id: live.id};
    pendingConfirmAttempts += 1;
    confirmPendingOpen(live).then(function (outcome) {
      pendingConfirmInFlight = null;
      if (outcome === "retry") {
        if (pendingConfirmAttempts < pendingConfirmMaxAttempts()) {
          // The reservation is HELD across the retry backoff: the intent is
          // still live and must not be replaced by another handoff meanwhile.
          schedulePendingConfirmRetry(live);
        } else {
          failPendingConfirmation(live);
          holdIncomingHandoffUntilExpiry(live);
        }
        return;
      }
      releaseIncomingHandoff(live);
      if (outcome === "standby") {
        // Not this page's destination (different UID or a proven different root):
        // release the incoming reservation so this unrelated page remains usable.
        pendingConfirmExhaustedId = live.id;
        return;
      }
      if (outcome === "failed") {
        // A definite record rejection keeps the hint + slot for later, but this
        // page does not retry it automatically; release its UI reservation.
        pendingConfirmExhaustedId = live.id;
        return;
      }
      // Confirmed or deliberately refused: release the correlated hint.
      consumedIntentId = live.id;
      incomingIntent = null;
      clearPendingOpen(live.id);
    }).catch(function () {
      pendingConfirmInFlight = null;
      if (pendingConfirmAttempts < pendingConfirmMaxAttempts()) {
        schedulePendingConfirmRetry(live);
      } else {
        failPendingConfirmation(live);
        holdIncomingHandoffUntilExpiry(live);
      }
    });
  }

  function setHandoffControlsDisabled(disabled) {
    // UX reinforcement of the single-flight flag (round-3 R3-1). The id
    // collision on `switch-study` means getElementById cannot be trusted to
    // return the BUTTON, so select the button elements themselves.
    try {
      if (typeof document.querySelectorAll !== "function") return;
      document.querySelectorAll('button#switch-study, .recent-reopen-button').forEach(function (btn) {
        btn.disabled = disabled;
      });
    } catch (_) {}
  }

  function beginHandoff() {
    // Round-3 R3-1: Switch and Reopen share ONE cookie slot. Starting a
    // second handoff while one is outstanding would let the older action's
    // refusal delete the newer intent (its continuation can only know its
    // own id), so overlapping actions are refused before any cookie is
    // written and before any second invoke.
    //
    // Review 5566059819 P2-2: before refusing, drop a reservation whose hint
    // is already gone, expired, or replaced (correlated release only) so the
    // page cannot stay single-flighted forever when timers were throttled.
    if (handoffOutstanding) releaseStaleIncomingClaim();
    if (handoffOutstanding) return false;
    handoffOutstanding = true;
    setHandoffControlsDisabled(true);
    return true;
  }

  function endHandoff() {
    handoffOutstanding = false;
    setHandoffControlsDisabled(false);
  }

  function clearPendingOpen(expectedId) {
    // Correlated clear: given an id, only drop the intent that id names, so
    // an older action can never discard a newer one (round-2 P1).
    try {
      if (expectedId) {
        const current = readPendingIntent();
        if (current && current.id !== expectedId) return;
      }
      document.cookie = PENDING_OPEN_COOKIE + "=; path=/; max-age=0; SameSite=Lax";
    } catch (_) {}
  }

  function discardIncomingIntent(expectedId) {
    // Drop the intent captured at this page's init and make it ineligible.
    // The ordinary binding paths pass their live id so a cookie that changed
    // in the meantime is left alone (round-3 R3-1); the restored-page
    // `#switch-failed` path passes nothing and force-clears deliberately.
    //
    // PR3 review 5566059819 P2-1: discarding the captured intent must also
    // release the CORRELATED reservation, or a restored `#switch-failed` page
    // (failed switch, zero records) stays permanently single-flighted with its
    // controls disabled. Only the claim correlated to the discarded id is
    // released; another action's active handoff is never ended, and the shared
    // cookie/slot are handled exactly as before.
    const discardedId = expectedId || (incomingIntent && incomingIntent.id) || incomingHandoffClaim;
    incomingIntent = null;
    clearPendingOpen(expectedId);
    if (discardedId && incomingHandoffClaim === discardedId) {
      incomingHandoffClaim = null;
      clearIncomingClaimExpiry();
      endHandoff();
    }
  }

  function maybeConsumePendingOpen() {
    if (consumedIntentId) return;                 // confirmed exactly once per page
    if (pendingConfirmInFlight || pendingConfirmTimer) return; // single-flight
    if (pendingConfirmExhaustedId) return;        // terminal here; a reload retries
    const atLoad = incomingIntent;
    if (!atLoad) return;                          // nothing eligible at init
    const live = readPendingIntent();
    if (!live || live.id !== atLoad.id) {
      // Gone, expired, or replaced: this page's correlation no longer protects
      // anything, so release the reservation it owns (review 5566059819 P2-2).
      // A live claim for the SAME id is never released here, and no other
      // action's handoff is touched.
      releaseStaleIncomingClaim();
      return;
    }
    if (live.producer === window.location.host) return; // producer, or its reload
    // Binding: only the backend actually bound to the target study records.
    // The full entry comes from the server-side pending slot (never from the
    // cookie), and the slot is confirmed + cleared only on a bound match.
    if (!state) return;                           // binding not known yet: retry
    if (typeof state.study_uid !== "string" || !state.study_uid) {
      discardIncomingIntent(live.id);            // cannot verify -> fail closed
      return;
    }
    // Bound destination: the pending slot names the target study; confirm it
    // (record same-origin once the bound study matches) or drop it. A
    // transient read failure keeps the hint and retries with the same id.
    startPendingConfirmation(live);
  }

  // Reserve an eligible incoming intent synchronously at page initialization,
  // before /api/studies or the first /state response can complete. The recent
  // list and Switch controls render independently; otherwise an early C request
  // can replace B's hint + only durable slot while B's state is delayed.
  incomingIntent = readPendingIntent();
  if (incomingIntent && incomingIntent.producer !== window.location.host) {
    claimIncomingHandoff(incomingIntent);
  }
  const RECENT_LABELS = {
    "ready": "Ready to reopen",
    "source-missing": "Source folder missing",
    "staged-missing": "Staged copy missing (will re-stage on open)",
    "unsafe-entry": "Cannot reopen (unsafe entry)"
  };

  function mountRecentAffordance() {
    if (typeof document.querySelector !== "function") return;
    const browser = document.querySelector("aside.series-browser");
    if (!browser) return;
    if (document.getElementById("recent-cases-section")) return;

    const section = document.createElement("section");
    section.id = "recent-cases-section";
    section.className = "recent-cases";

    const heading = document.createElement("h2");
    heading.textContent = "Recent Cases";
    section.appendChild(heading);

    const banner = document.createElement("div");
    banner.id = "recent-banner";
    banner.className = "recent-banner error";
    banner.setAttribute("role", "alert");
    banner.style.display = "none";
    section.appendChild(banner);

    const list = document.createElement("ul");
    list.id = "recent-list";
    list.className = "recent-list";
    section.appendChild(list);

    browser.appendChild(section);
  }

  function showRecentBanner(text) {
    const banner = document.getElementById("recent-banner");
    if (!banner) return;
    banner.textContent = text;
    banner.style.display = "block";
  }

  function hideRecentBanner() {
    const banner = document.getElementById("recent-banner");
    if (!banner) return;
    banner.textContent = "";
    banner.style.display = "none";
  }

  // PR3 review r4311831218: the "current case" marker is RECONCILED state, not
  // a one-time snapshot taken while the list happens to render. Two independent
  // facts define "current", and either may arrive AFTER the rows exist:
  //   (1) the independently established bound study identity (`boundStudyUid`,
  //       from the backend's own state projection), and
  //   (2) the row's SERVER-RESOLVED destination equal to the bound root
  //       (`currentLiveRoot()`, from /api/library/bound-root) — a bare study
  //       UID match is not enough, because the same study restaged under a
  //       different stage base is a different workspace and Reopen already
  //       requires a switch for it.
  // Until both are known a row is NOT current (unknown is never "current"), and
  // every trigger that can establish a fact re-evaluates all rendered rows.
  let recentMarkerRows = [];

  function reconcileRecentMarkers() {
    const boundStudy = typeof boundStudyUid === "string" && boundStudyUid ? boundStudyUid : null;
    const liveRoot = currentLiveRoot();
    recentMarkerRows.forEach(function (row) {
      if (!row.item) return;
      const entryStudy = typeof row.entry.studyUid === "string" && row.entry.studyUid
        ? row.entry.studyUid
        : null;
      const entryDest = canonicalRootString(row.probedDestRoot);
      const isCurrent = Boolean(
        boundStudy && entryStudy && entryStudy === boundStudy &&
        liveRoot && entryDest && entryDest === liveRoot
      );
      row.item.classList.toggle("active", isCurrent);
      // PR3 review r4313043501: the marker must also be VISIBLE on the served
      // profile, not only class/attribute state. The label is owned by this same
      // reconciler, so a row that stops being current cannot keep a visible
      // "current" claim.
      if (row.currentLabel) row.currentLabel.hidden = !isCurrent;
      if (isCurrent) row.item.setAttribute("aria-current", "true");
      else row.item.removeAttribute("aria-current");
    });
  }

  async function loadRecentCases() {
    const generation = ++recentListGeneration;
    hideRecentBanner();
    let response;
    try {
      response = await fetch("/api/library/recent");
    } catch (err) {
      if (generation !== recentListGeneration) return;
      showRecentBanner("Failed to connect to recent cases service: " + err.message);
      return;
    }
    if (!response.ok) {
      let msg = "HTTP " + response.status;
      try {
        const errJson = await response.json();
        if (errJson && errJson.message) msg = errJson.message;
      } catch (_) {}
      if (generation !== recentListGeneration) return;
      showRecentBanner("Recent cases error: " + msg);
      return;
    }
    const data = await response.json();
    if (generation !== recentListGeneration) return;  // superseded/abandoned: never publish
    recentDigest = data.digest || null;
    const list = document.getElementById("recent-list");
    if (!list) return;
    list.replaceChildren();
    // Replaced rows are detached: drop them so no stale element is reconciled.
    recentMarkerRows = [];

    if (data.reason === "corrupt-rows") {
      showRecentBanner("Recent cases store contains corrupt rows; entries degraded to empty (tamper detected).");
      return;
    }

    const entries = Array.isArray(data.entries) ? data.entries : [];
    if (entries.length === 0) {
      const emptyLi = document.createElement("li");
      emptyLi.className = "recent-empty-item hint";
      emptyLi.textContent = "No recent cases.";
      list.appendChild(emptyLi);
      return;
    }

    for (const entry of entries) {
      const item = document.createElement("li");
      item.className = "recent-item";
      // PR3 review r4311831218: no one-time UID-only snapshot here. The row is
      // registered with the reconciler and marked only once the bound study
      // identity AND the row's server-resolved destination are both known.
      const markerRow = {entry: entry, item: item, probedDestRoot: null};
      recentMarkerRows.push(markerRow);

      const infoSpan = document.createElement("span");
      infoSpan.className = "recent-info";
      infoSpan.textContent = entry.studyUid + " (opened: " + entry.openCount + ")";
      item.appendChild(infoSpan);

      const statusBadge = document.createElement("span");
      statusBadge.className = "recent-badge";
      statusBadge.textContent = "Checking...";
      item.appendChild(statusBadge);

      // PR3 review r4313043501: the sighted-user affordance for the reconciled
      // "current case" state. Hidden until the reconciler proves the row current
      // (bound study identity AND this row's server-resolved destination equal
      // the live root), and hidden again the moment either fact changes.
      const currentLabel = document.createElement("span");
      currentLabel.className = "recent-current-label";
      currentLabel.textContent = "Current case";
      currentLabel.hidden = true;
      item.appendChild(currentLabel);
      markerRow.currentLabel = currentLabel;

      // Async availability probe. The probe's `dest_root` is the server's
      // own resolved destination derivation for this entry — authoritative
      // no-op input when it has answered (F2).
      //
      // Bounded probe wait (R11 review finding): the Reopen decision must
      // await server-resolved destination verification before evaluating
      // same-root identity. An early click while the probe is pending must
      // NOT fall back to client lexical derivation (which differs on path
      // variants such as /stage/./) and respawn the active workspace.
      let probeStatus = "idle"; // "idle" | "pending" | "settled" | "timed_out" | "failed"
      let probedDestRoot = null;
      let probeReady = null;
      let probeAttempt = 0;

      function loadEntryProbe() {
        // Same bounded server-resolve discipline as the pending-open
        // confirmation (shared fetchResolvedDestRoot): the deadline spans
        // fetch AND body consumption, and a superseded/timed-out attempt can
        // never publish a destination or overwrite a newer status.
        probeStatus = "pending";
        const thisAttempt = ++probeAttempt;
        return fetchResolvedDestRoot(entry, function () {
          return thisAttempt !== probeAttempt;
        }).then(function (result) {
          if (result.status === "stale") return null;
          if (result.status === "settled") {
            const probeData = result.data || {};
            const avail = probeData.availability || "unsafe-entry";
            statusBadge.textContent = RECENT_LABELS[avail] || avail;
            statusBadge.dataset.availability = avail;
            if (result.destRoot) probedDestRoot = result.destRoot;
            probeStatus = "settled";
            // PR3 review r4311831218: the row's server-resolved destination is
            // now known — re-evaluate this row's current-case marker.
            markerRow.probedDestRoot = probedDestRoot;
            reconcileRecentMarkers();
            return probedDestRoot;
          }
          probeStatus = result.status;
          statusBadge.textContent = result.status === "timed_out"
            ? "Cannot reopen (probe timed out)"
            : "Cannot reopen (probe failed)";
          return null;
        });
      }

      function ensureEntryProbe() {
        if (!probeReady) {
          probeReady = Promise.resolve().then(function () {
            return loadEntryProbe();
          });
        }
        return probeReady;
      }

      // Start probe on render
      ensureEntryProbe();

      const reopenBtn = document.createElement("button");
      reopenBtn.type = "button";
      reopenBtn.className = "recent-reopen-button";
      reopenBtn.textContent = "Reopen";
      reopenBtn.addEventListener("click", async function () {
        // Same-root no-op (review P2, S10 `unchanged` leg): reopening the
        // already-open root must NOT kill/respawn the backend — the shell
        // has no same-root no-op (main.rs always terminates), so the page
        // refuses the request before any handoff, cookie, or invoke. The
        // no-op requires ACTUAL same-root identity: the entry's resolved
        // destination must equal the live root — a bare StudyInstanceUID
        // match is NOT sufficient (the same study restaged under a new
        // stage base is a different root and must switch, never no-op).
        // Wait for the bound-root initialization to settle first: the recent
        // list renders independently of it, so an early click must never
        // decide from a still-null live root (that shape kills/respawns the
        // bound backend for a no-op). Settled-null keeps the no-op off.
        await boundRootInitialized();
        if (boundRootStatus === "timed_out" || boundRootStatus === "failed") {
          // Bounded initialization wait (review finding): if the request
          // stalled or timed out, leave the current backend untouched (no
          // invoke, no cookie, no queue write) and surface an actionable
          // retry/refusal notice. Reset memoized promise so the user can retry.
          boundRootReady = null;
          show(
            "Cannot verify active workspace (server initialization " +
            (boundRootStatus === "timed_out" ? "timed out" : "failed") +
            "); retry Reopen or reload the page.",
            true
          );
          return;
        }

        // Bounded entry probe wait (R11 review finding F2):
        // Authoritative destination resolution MUST be known before deciding
        // whether this entry equals the live root. An un-probed entry must
        // NOT fall back to client lexical derivation (which may mismatch
        // anchored equivalents such as /stage/./) and respawn the active workspace.
        await ensureEntryProbe();
        if (probeStatus === "timed_out" || probeStatus === "failed" || !probedDestRoot) {
          probeReady = null;
          show(
            "Cannot verify case destination (probe " +
            (probeStatus === "timed_out" ? "timed out" : "failed") +
            "); retry Reopen or reload the page.",
            true
          );
          return;
        }

        // PR3 review r4310770076: the two path checks above prove only that
        // the paths agree — never WHICH study the bound backend serves. Wait
        // (bounded) for the independently established bound study identity and
        // require it to equal the entry's studyUid before answering "already
        // open". An unknown or mismatching identity is never reported as
        // already open: the click falls through to the shell, whose prove step
        // owns real mismatch rejection (a false no-op would leave a different
        // study on screen while claiming the entry is open).
        await boundIdentityInitialized();
        const liveRoot = currentLiveRoot();
        // Authoritative destination: the server probe's resolved derivation
        // is now guaranteed established. The persisted display-only destRoot
        // is NEVER no-op authority (F2), and the live root is the server-reported
        // bound root (F5).
        const entryDest = canonicalRootString(probedDestRoot);
        const entryStudy = typeof entry.studyUid === "string" && entry.studyUid ? entry.studyUid : null;
        const boundStudy = typeof boundStudyUid === "string" && boundStudyUid ? boundStudyUid : null;
        if (
          entryDest && liveRoot && entryDest === liveRoot &&
          entryStudy && boundStudy && entryStudy === boundStudy
        ) {
          show("Case " + entry.studyUid + " is already open; nothing to reopen.", false);
          return;
        }
        const invoke = switchBridge();
        if (!invoke) {
          show("Reopening requires desktop shell (runtime-002 target binding). In browser fallback, target binding cannot be executed.", true);
          return;
        }
        const args = {
          source_root: entry.scanSourcePath,
          stage_base: entry.stageBaseDir,
          study_uid: entry.studyUid,
          sourceRoot: entry.scanSourcePath,
          stageBase: entry.stageBaseDir,
          studyUid: entry.studyUid
        };
        show("Reopen requested for study: " + entry.studyUid, false);
        // Handoff (review P1-4): queue BEFORE invoke — A is dead and the
        // page navigates before the command returns (see setPendingOpen).
        // Round-3 R3-1: reopen shares the single handoff slot with switch,
        // and clears only the id it created.
        if (!beginHandoff()) {
          show("A switch or reopen is already in progress; wait for it to finish.", true);
          return;
        }
        const intentId = setPendingOpen(entry);
        rememberActiveHandoff(intentId, entry);
        // QUEUE-011 (reopen leg): same single-awaited-queue discipline as
        // the switch leg above — one queue POST, awaited before invoke,
        // rejection-tolerant.
        // Owner-ratified 2026-10-15: ANY queue error, timeout, or non-OK
        // write arms the explicit "may not be saved" notice (applied after
        // outcome handling below). Best-effort preserved: a queue failure
        // still never breaks the reopen itself.
        let queueNoticeArmed = false;
        try {
          if (!await queuePendingOpen(intentId, Date.now() + 120000, entry)) queueNoticeArmed = true;
        } catch (_) {
          queueNoticeArmed = true;
        }
        // F4: same cross-navigation notice discipline as the switch leg.
        if (queueNoticeArmed) armRecentNotice("queue-error");
        invoke("switch_study", args).then(function (result) {
          let outcome = result;
          if (typeof outcome === "string") {
            try { outcome = JSON.parse(outcome); } catch (_) {}
          }
          const isNeedsStage = (outcome && typeof outcome === "object" && outcome.status === "needs_stage") ||
                               (typeof outcome === "string" && outcome.indexOf("needs_stage:") === 0);
          if (isNeedsStage) {
            let destPath = "";
            if (outcome && typeof outcome === "object" && outcome.dest_root) {
              destPath = outcome.dest_root;
            } else if (typeof outcome === "string") {
              const rest = outcome.slice("needs_stage:".length);
              const lastColon = rest.lastIndexOf(":");
              destPath = (lastColon > 1) ? rest.slice(0, lastColon) : rest;
            }
            return fetch("/api/library/stage", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({
                source_path: args.source_root,
                study_uid: args.study_uid,
                dest_path: destPath
              })
            }).then(function (res) {
              if (!res.ok) throw new Error("stage request failed (" + res.status + ")");
              return res.json();
            }).then(function () {
              // PR3 review 5564276070 F2: renew the correlated intent before the
              // final reopen invoke (same id; never overwrite a newer handoff).
              return renewPendingOpen(intentId, entry).then(function (renewed) {
                if (!renewed) {
                  queueNoticeArmed = true;
                  armRecentNotice("queue-error");
                }
                return true; // best-effort reopen, but warn before navigation
              });
            }).then(function () {
              return invoke("switch_study", args);
            });
          }
          return outcome;
        }).then(function (finalResult) {
          // Contract rule: only the SUCCESSFULLY opened identity is recorded
          // (moves front + accumulates openCount). A refused/failed switch
          // must never touch the recent document.
          if (finalResult && typeof finalResult === "object" && finalResult.status === "switched") {
            dismissRecoveryBanner();
            // Handoff (review P1-4): do NOT POST here — backend A is already
            // dead and the shell has navigated. The pending cookie queued
            // before invoke survives the navigation; the bound destination
            // page evaluates it (maybeConsumePendingOpen) and records
            // same-origin once its bound study matches the target.
            // C2: track the granted root for the same-root no-op.
            if (typeof finalResult.dest_root === "string") noteLiveRoot(finalResult.dest_root);
            releaseActiveHandoff(intentId);
            endHandoff();
            return null;
          }
          clearPendingOpen(intentId);
          releaseActiveHandoff(intentId);
          endHandoff();
          return null;
        }).catch(function (error) {
          clearPendingOpen(intentId);
          releaseActiveHandoff(intentId);
          endHandoff();
          show("Reopen refused: " + error, true);
        }).then(function () {
          // R5: re-apply the queue notice AFTER outcome handling so the
          // terminal "Reopen refused" line cannot silently erase the
          // owner-required "may not be saved" labeling. (On the switched
          // path no later show() runs, so this appends the notice to the
          // pre-invoke message state.)
          if (queueNoticeArmed) {
            queueNoticeArmed = false;
            show(((message.textContent) ? message.textContent + " " : "") + QUEUE_TIMEOUT_NOTICE, true);
          }
        });
      });

      item.appendChild(reopenBtn);
      list.appendChild(item);
    }
    // Rows whose two facts were ALREADY known at render time are marked here;
    // late facts re-enter through reconcileRecentMarkers().
    reconcileRecentMarkers();
  }

  async function recordRecentCase(entry, openId) {
    const payload = {
      version: 1,
      entry: {
        studyUid: entry.studyUid,
        seriesUid: entry.seriesUid || null,
        scanSourcePath: entry.scanSourcePath,
        stageBaseDir: entry.stageBaseDir,
        destRoot: entry.destRoot || "",
        openedSeq: 0,
        openCount: 1
      },
      base_digest: recentDigest
    };
    // Replay idempotence (review P2): the pending intent id doubles as the
    // server-side open id, so a transport replay of THIS record delivery is
    // deduplicated instead of counted twice — including the bounded retry after
    // an UNKNOWN commit.
    if (typeof openId === "string" && openId) payload.open_id = openId;

    // One coherent acknowledgement deadline (PR3 review 5564032012 F2). The
    // budget spans: POST headers + body -> (409) stale body + retry POST
    // headers + body -> ack body. The exchange below is PURE (it never writes
    // shared state), so a late settlement of a timed-out attempt can never
    // publish a digest, record twice, or masquerade as a commit.
    const deadlineMs = recordAckTimeoutMs();
    let deadlineTimer = null;
    let timedOut = false;
    let controller = null;
    if (typeof AbortController !== "undefined") {
      controller = new AbortController();
    }
    const signal = controller ? controller.signal : undefined;
    const deadline = new Promise(function (_, reject) {
      deadlineTimer = setTimeout(function () {
        timedOut = true;
        if (controller) {
          try { controller.abort(); } catch (_) {}
        }
        const timeoutErr = new Error("Recent case record acknowledgement timed out");
        timeoutErr.code = "record-unknown";
        reject(timeoutErr);
      }, deadlineMs);
    });

    const post = async function () {
      const opts = {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      };
      if (signal) opts.signal = signal;
      return fetch("/api/library/recent", opts);
    };

    const exchange = (async function () {
      let response = await post();
      if (response.status === 409) {
        // Stale snapshot: reload and retry with merge — both inside the SAME
        // coherent deadline as the first attempt.
        const curData = await response.json();
        payload.base_digest = curData.current ? curData.current.digest : null;
        response = await post();
      }
      if (!response.ok) {
        // A definite rejection is NOT an unknown commit: report the failure
        // (and let the caller stop rather than retry a refused write).
        const status = response.status;
        const rejectErr = new Error("Recent case save failed: HTTP " + status);
        rejectErr.code = "record-rejected";
        rejectErr.status = status;
        throw rejectErr;
      }
      const saved = await response.json();
      return saved && typeof saved.digest === "string" ? saved.digest : null;
    })();
    // Late settlement of a timed-out exchange is isolated: the race below has
    // already decided, and this handler only keeps a late rejection from
    // surfacing as unhandled.
    exchange.catch(function () {});

    let digest;
    try {
      digest = await Promise.race([exchange, deadline]);
    } catch (err) {
      if (err && err.code === "record-rejected") {
        // Definite rejection: the server answered non-OK, so the write was
        // refused. Surface it and let the caller stop — a refused write is never
        // retried.
        show("Recent case save failed: HTTP " + err.status, true);
        throw err;
      }
      // PR3 review 5564147464 (app.js:1945): EVERY other failure here is
      // INDETERMINATE, not a rejection — an immediate fetch rejection
      // (connection reset before/while the request was sent), an
      // aborted/truncated acknowledgement body (the server may already have
      // committed), or the coherent deadline. Normalize them all to
      // record-unknown so the caller spends its bounded retry budget with the
      // SAME idempotent open id instead of treating a transient transport
      // failure as a lost save.
      const unknownErr = new Error("Recent case record outcome unknown");
      unknownErr.code = "record-unknown";
      unknownErr.cause = err;
      throw unknownErr;
    } finally {
      if (deadlineTimer !== null) {
        try { clearTimeout(deadlineTimer); } catch (_) {}
      }
    }
    if (digest !== null) recentDigest = digest;

    // KNOWLEDGE state: the ack is KNOWN. Ancillary work must never block the
    // confirmed outcome, and a hung refresh/cleanup must never be reported as
    // an unknown commit (PR3 review 5564032012 F2): the refresh is bounded,
    // detached, and swallowed.
    refreshRecentListBounded();
  }

  function refreshRecentListBounded() {
    // Detached, bounded post-save refresh: it can never block a KNOWN record
    // acknowledgement, and its failure or hang is irrelevant to commit state.
    const timeoutMs = pendingCleanupTimeoutMs();
    let timer = null;
    let controller = null;
    if (typeof AbortController !== "undefined") {
      controller = new AbortController();
    }
    const refresh = loadRecentCases();
    refresh.catch(function () {});
    return Promise.race([
      refresh.catch(function () {}),
      new Promise(function (resolve) {
        timer = setTimeout(function () {
          if (controller) {
            try { controller.abort(); } catch (_) {}
          }
          resolve(null);
        }, timeoutMs);
      })
    ]).then(function () {
      if (timer !== null) {
        try { clearTimeout(timer); } catch (_) {}
      }
      return null;
    }, function () {
      if (timer !== null) {
        try { clearTimeout(timer); } catch (_) {}
      }
      return null;
    });
  }

  function clearPendingSlotBounded(id) {
    // Bounded pending-slot cleanup (PR3 review 5564032012 F2): a hung clear
    // acknowledgement can never block the confirmed outcome or be mistaken for
    // an unknown commit. The durable slot expires by its own `exp` and a later
    // replay is deduplicated by the persistent open id.
    const timeoutMs = pendingCleanupTimeoutMs();
    let timer = null;
    let controller = null;
    if (typeof AbortController !== "undefined") {
      controller = new AbortController();
    }
    const signal = controller ? controller.signal : undefined;
    const opts = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({clear: true, id: id})
    };
    if (signal) opts.signal = signal;
    const clear = fetch("/api/library/recent/pending", opts);
    clear.catch(function () {});
    return Promise.race([
      clear.catch(function () {}),
      new Promise(function (resolve) {
        timer = setTimeout(function () {
          if (controller) {
            try { controller.abort(); } catch (_) {}
          }
          resolve(null);
        }, timeoutMs);
      })
    ]).then(function () {
      if (timer !== null) {
        try { clearTimeout(timer); } catch (_) {}
      }
      return null;
    }, function () {
      if (timer !== null) {
        try { clearTimeout(timer); } catch (_) {}
      }
      return null;
    });
  }

  // Restored-A notice: after a post-kill recovery the shell re-navigates to
  // the rebound authority with `#switch-failed`, so the restored page never
  // silently looks like a successful switch (no silent failure; the fragment
  // never reaches the server and is cleared after being shown once).
  // The notice is rendered into a persistent banner that normal initial series loading
  // (chooseSeries -> show("Series loaded", false)) cannot erase until dismissed
  // or a subsequent switch succeeds.
  if (window.location && window.location.hash === "#switch-failed") {
    // Restored-A is a FAILED switch: discard any queued first-open intent
    // before startup consumption runs (this block executes synchronously
    // during script eval, ahead of the async consume at startup).
    discardIncomingIntent();
    showRecoveryBanner("The study switch did not complete: the previous study was restored.");
    show("The study switch did not complete: the previous study was restored.", true);
    if (window.history && typeof window.history.replaceState === "function") {
      window.history.replaceState(null, "", window.location.pathname + window.location.search);
    }
  }

  document.getElementById("play").addEventListener("click", function () { setPlaying(!playing); });
  document.getElementById("next-frame").addEventListener("click", function () { moveFrame(1); });
  document.getElementById("previous-frame").addEventListener("click", function () { moveFrame(-1); });
  document.getElementById("next-slice").addEventListener("click", function () { moveSlice(1); });
  document.getElementById("previous-slice").addEventListener("click", function () { moveSlice(-1); });
  document.getElementById("fps").addEventListener("input", function (event) {
    fps = Number(event.target.value);
    document.getElementById("fps-value").textContent = String(fps);
    loadBundle(false).catch(function (error) { show(error.message, true); });
  });
  document.getElementById("mark-ed").addEventListener("click", function () {
    edFrame = frame;
    sendAction("select_ed_es", {ed_frame: edFrame, es_frame: esFrame === null ? frame : esFrame}).catch(function (error) { show(error.message, true); });
  });
  document.getElementById("mark-es").addEventListener("click", function () {
    esFrame = frame;
    sendAction("select_ed_es", {ed_frame: edFrame === null ? frame : edFrame, es_frame: esFrame}).catch(function (error) { show(error.message, true); });
  });
  document.getElementById("apply-inclusion").addEventListener("click", function () {
    sendAction("apply_inclusion_and_refresh", {}).catch(function (error) { show(error.message, true); });
  });
  document.getElementById("reset-inclusion").addEventListener("click", function () {
    if (state) {
      const resetDraft = currentDraft() || [];
      resetDraft.forEach(function (row) { row.included = true; });
      draftLineage = updateDraftLineage(
        draftLineage,
        {type: "reset", state: state, draft: resetDraft}
      );
      syncDraftProjection();
    }
    renderGrid();
    loadBundle(false).catch(function (error) { show(error.message, true); });
  });
  document.getElementById("reference-sex").addEventListener("change", function () {
    updateReportLinks();
    loadBundle(false).catch(function (error) { show(error.message, true); });
  });
  document.getElementById("wl-apply").addEventListener("click", applyWindowLevel);
  document.getElementById("wl-reset").addEventListener("click", resetWindowLevel);
  document.getElementById("visibility-reset").addEventListener("click", resetVisibility);
  document.getElementById("canvas-wrap").addEventListener("wheel", function (event) {
    event.preventDefault();
    zoom = Math.max(0.5, Math.min(3, zoom + (event.deltaY < 0 ? 0.1 : -0.1)));
    canvas.style.transform = "scale(" + zoom + ")";
  }, {passive: false});
  document.getElementById("shortcuts").addEventListener("click", function () { document.getElementById("shortcuts-dialog").showModal(); });
  document.getElementById("close-shortcuts").addEventListener("click", function () { document.getElementById("shortcuts-dialog").close(); });
  document.addEventListener("keydown", function (event) {
    if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
    const command = window.CMR_KEYBOARD_SHORTCUTS[event.key];
    if (!command) return;
    event.preventDefault();
    if (command === "toggle_playback") setPlaying(!playing);
    else if (command === "step_forward") moveFrame(1);
    else if (command === "step_backward") moveFrame(-1);
    else if (command === "next_slice") moveSlice(1);
    else if (command === "previous_slice") moveSlice(-1);
    else if (command === "increase_speed") {
      fps = Math.min(60, fps + 1);
      document.getElementById("fps").value = String(fps);
      loadBundle(false).catch(function (error) { show(error.message, true); });
    }
    else if (command === "decrease_speed") {
      fps = Math.max(1, fps - 1);
      document.getElementById("fps").value = String(fps);
      loadBundle(false).catch(function (error) { show(error.message, true); });
    }
    else if (command === "mark_ed") document.getElementById("mark-ed").click();
    else if (command === "mark_es") document.getElementById("mark-es").click();
    else if (command === "toggle_shortcuts") document.getElementById("shortcuts").click();
  });

  window.setInterval(function () {
    if (seriesUid) loadState().catch(function (error) { show(error.message, true); });
  }, 1000);
  mountRecentAffordance();
  // F5: learn the genuinely bound root on EVERY initialization, independent
  // of the recent store and of any pending intent. The memoized promise is
  // what Reopen decisions await (delayed bound-root review fix).
  boundRootInitialized();
  // The list load is STARTED here but the notice delivery below never awaits
  // it (an async function never throws synchronously into this sequence).
  const initialRecentLoad = loadRecentCases().catch(function (error) {
    show("Recent cases load failed: " + error.message, true);
  });
  // PR3 review r4310770084: deliver the cross-navigation queue-failure notice
  // INDEPENDENTLY of the recent-list load. That load has no deadline around its
  // fetch or body consumption, so a stalled recent GET must not withhold the
  // "may not be saved" warning (loadStudies settles on its own). The call sits
  // AFTER the load has been started so the load's synchronous banner reset
  // cannot wipe the delivered warning banner, and still BEFORE the pending-open
  // consumption below, whose record-failure messages may follow it.
  consumeRecentNotice();
  initialRecentLoad.then(function () {
    // Bound destination of a successful switch records its pending open
    // here (review P1-4); `#switch-failed` restored pages already discarded
    // the intent synchronously above, so they never record. Retry is driven
    // by loadState, which fires once the bound study becomes known.
    maybeConsumePendingOpen();
  });
  loadStudies().catch(function (error) { show(error.message, true); });
  Object.assign(window.CMR_WORKSTATION, {
    loadStudies: loadStudies,
    loadBundle: loadBundle,
    getFrameRequestCount: function () { return frameRequestCount; },
    loadRecentCases: loadRecentCases,
    recordRecentCase: recordRecentCase,
    mountRecentAffordance: mountRecentAffordance,
    noteLiveRoot: noteLiveRoot,
    currentLiveRoot: currentLiveRoot
  });
  // Harness-only hook for the C2 same-root witnesses: replays the live-root
  // assignment a real `switched` outcome would perform. Production code
  // never reads this hook — the page learns the root only from genuine
  // shell outcomes above.
  window.__cmrTestHooks = {
    noteLiveRoot: noteLiveRoot,
    currentLiveRoot: currentLiveRoot,
    // Harness-only hook (PR3 review F3): re-evaluate the pending-open handoff
    // without a live interval, so the single-flight / bounded-retry behaviour
    // is directly exercisable. Production code never reads this hook.
    maybeConsumePendingOpen: maybeConsumePendingOpen,
    // Harness-only hook (review 5566059819 P2-2): observe the incoming
    // reservation directly, so claim-armed expiry cleanup is witnessed even
    // when no click happens. Production code never reads this hook.
    incomingReservationHeld: function () { return incomingHandoffClaim !== null; }
  };
}());
