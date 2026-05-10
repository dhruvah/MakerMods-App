"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Check,
  ChevronRight,
  CircleDot,
  Clock,
  Layers,
  Play,
  RotateCcw,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { services } from "@/lib/services";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatTime(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function phaseLabel(phase: string): string {
  if (phase === "init") return "Initializing";
  return phase
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// ── Component ────────────────────────────────────────────────────────────────

interface ActStatus {
  phase: string;
  elapsed_s: number;
  paused: boolean;
  available_phases: string[];
}

export function ActStep() {
  const { state, dispatch } = useWizard();

  const [status, setStatus] = useState<ActStatus | null>(null);
  const [elapsedS, setElapsedS] = useState(0);
  const [advancing, setAdvancing] = useState<string | null>(null);
  const [newPhase, setNewPhase] = useState("");
  const [showAddPhase, setShowAddPhase] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastPhaseRef = useRef<string>("init");

  // ── Polling ────────────────────────────────────────────────────────────────

  const fetchStatus = useCallback(async () => {
    try {
      const data = await services.getActStatus();
      setStatus(data);
      setError(null);
      // Sync timer when phase changes (backend is authoritative on elapsed_s)
      if (data.phase !== lastPhaseRef.current) {
        lastPhaseRef.current = data.phase;
        setElapsedS(data.elapsed_s);
      }
    } catch {
      setError("Cannot reach backend");
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 3000);
    return () => clearInterval(id);
  }, [fetchStatus]);

  // ── Client-side count-up timer ─────────────────────────────────────────────

  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setElapsedS((prev) => prev + 1);
    }, 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [status?.phase]);

  // ── Phase actions ──────────────────────────────────────────────────────────

  async function handleAdvance(phaseName: string) {
    setAdvancing(phaseName);
    try {
      await services.actAdvancePhase(phaseName);
      setElapsedS(0);
      lastPhaseRef.current = phaseName;
      setStatus((prev) => prev ? { ...prev, phase: phaseName, elapsed_s: 0 } : prev);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to advance phase");
    } finally {
      setAdvancing(null);
    }
  }

  async function handleReset() {
    try {
      await services.actReset();
      setElapsedS(0);
      lastPhaseRef.current = "init";
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to reset");
    }
  }

  async function handleAddPhase() {
    const trimmed = newPhase.trim().toLowerCase().replace(/\s+/g, "_");
    if (!trimmed) return;
    const updated = [...(status?.available_phases ?? state.actPhases), trimmed];
    try {
      await services.actSetPhases(updated);
      dispatch({ type: "SET_ACT_PHASES", phases: updated });
      setNewPhase("");
      setShowAddPhase(false);
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add phase");
    }
  }

  async function handleRemovePhase(phase: string) {
    const updated = (status?.available_phases ?? state.actPhases).filter((p) => p !== phase);
    if (updated.length === 0) return;
    try {
      await services.actSetPhases(updated);
      dispatch({ type: "SET_ACT_PHASES", phases: updated });
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to remove phase");
    }
  }

  // ── Derived state ──────────────────────────────────────────────────────────

  const phases = status?.available_phases ?? state.actPhases;
  const currentPhase = status?.phase ?? "init";
  const activeIdx = phases.indexOf(currentPhase);

  return (
    <StepCard
      title="ACT Control"
      description="Monitor phase progression and manually override the active ACT policy phase."
      showNext={false}
    >
      <div className="space-y-6">

        {/* ── Current phase + timer ── */}
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border bg-card p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
              Current Phase
            </p>
            <div className="flex items-center gap-2">
              <CircleDot
                className={cn(
                  "h-4 w-4 shrink-0",
                  currentPhase === "init"
                    ? "text-muted-foreground"
                    : "text-blue-500 animate-pulse"
                )}
              />
              <span className="text-lg font-semibold truncate">
                {phaseLabel(currentPhase)}
              </span>
            </div>
          </div>

          <div className="rounded-lg border bg-card p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
              <Clock className="inline h-3 w-3 mr-1 -mt-0.5" />
              Elapsed
            </p>
            <span className="text-2xl font-mono font-semibold tabular-nums">
              {formatTime(elapsedS)}
            </span>
          </div>
        </div>

        {/* ── Phase timeline ── */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-medium flex items-center gap-1.5">
              <Layers className="h-4 w-4" />
              Phase Sequence
            </p>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs gap-1"
              onClick={handleReset}
            >
              <RotateCcw className="h-3 w-3" />
              Reset
            </Button>
          </div>

          <div className="space-y-2">
            {phases.map((phase, i) => {
              const isActive = phase === currentPhase;
              const isDone = activeIdx > 0 && i < activeIdx;
              const isPending = !isActive && !isDone;
              const isAdvancing = advancing === phase;

              return (
                <div
                  key={phase}
                  className={cn(
                    "flex items-center gap-3 rounded-lg border px-4 py-3 transition-colors",
                    isActive && "border-blue-400 bg-blue-50 dark:border-blue-700 dark:bg-blue-950/40",
                    isDone && "border-transparent bg-muted/40",
                    isPending && "border-dashed border-muted-foreground/20"
                  )}
                >
                  {/* Step indicator */}
                  <div
                    className={cn(
                      "flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-medium",
                      isActive && "bg-blue-500 text-white",
                      isDone && "bg-primary text-primary-foreground",
                      isPending && "border-2 border-muted-foreground/30 text-muted-foreground"
                    )}
                  >
                    {isDone ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : isActive ? (
                      <ChevronRight className="h-3.5 w-3.5" />
                    ) : (
                      <span>{i + 1}</span>
                    )}
                  </div>

                  {/* Phase name */}
                  <div className="flex-1 min-w-0">
                    <p
                      className={cn(
                        "text-sm font-medium",
                        isPending && "text-muted-foreground"
                      )}
                    >
                      {phaseLabel(phase)}
                    </p>
                    <p className="text-xs text-muted-foreground font-mono">{phase}</p>
                  </div>

                  {/* Elapsed badge for active phase */}
                  {isActive && (
                    <Badge
                      variant="secondary"
                      className="font-mono text-xs shrink-0 bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                    >
                      {formatTime(elapsedS)}
                    </Badge>
                  )}

                  {/* Select button */}
                  <Button
                    size="sm"
                    variant={isActive ? "default" : "outline"}
                    className="shrink-0 h-7 text-xs"
                    disabled={isActive || isAdvancing !== null}
                    onClick={() => handleAdvance(phase)}
                  >
                    {isAdvancing ? (
                      <span className="flex items-center gap-1">
                        <Play className="h-3 w-3 animate-pulse" />
                        Starting…
                      </span>
                    ) : isActive ? (
                      "Active"
                    ) : (
                      "Select"
                    )}
                  </Button>

                  {/* Remove phase button */}
                  <button
                    type="button"
                    className="text-muted-foreground/40 hover:text-destructive transition-colors shrink-0"
                    title="Remove phase"
                    onClick={() => handleRemovePhase(phase)}
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Add phase ── */}
        {showAddPhase ? (
          <div className="flex items-end gap-2">
            <div className="flex-1 space-y-1.5">
              <Label htmlFor="new-phase">New Phase Name</Label>
              <Input
                id="new-phase"
                placeholder="e.g. place_bread"
                value={newPhase}
                onChange={(e) => setNewPhase(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAddPhase()}
                autoFocus
              />
              <p className="text-xs text-muted-foreground">
                Spaces are converted to underscores automatically.
              </p>
            </div>
            <Button onClick={handleAddPhase} disabled={!newPhase.trim()}>
              Add
            </Button>
            <Button variant="ghost" onClick={() => { setShowAddPhase(false); setNewPhase(""); }}>
              Cancel
            </Button>
          </div>
        ) : (
          <Button
            variant="outline"
            size="sm"
            className="w-full border-dashed"
            onClick={() => setShowAddPhase(true)}
          >
            + Add Phase
          </Button>
        )}

        <Separator />

        {/* ── Instructions ── */}
        <div className="rounded-lg bg-muted/50 px-4 py-3 text-xs text-muted-foreground space-y-1.5">
          <p className="font-medium text-foreground text-sm">How to use</p>
          <p>
            Start the orchestrator via the CLI:{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono">
              python -m orchestrator.start
            </code>
          </p>
          <p>
            Click <strong>Select</strong> to manually advance to any phase — this overrides
            the AI agent and immediately activates that phase.
          </p>
          <p>
            The elapsed timer resets each time a phase is activated. Use{" "}
            <strong>Reset</strong> to return state to <em>init</em>.
          </p>
        </div>

        {/* ── Error banner ── */}
        {error && (
          <div className="flex items-center gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
            <X className="h-4 w-4 shrink-0" />
            {error}
            <button
              type="button"
              className="ml-auto text-xs underline"
              onClick={() => setError(null)}
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
    </StepCard>
  );
}
