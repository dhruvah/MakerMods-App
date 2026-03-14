// Robot mode
export type RobotMode = "single" | "bimanual";

// Port info from backend
export interface PortInfo {
  port: string;
  description: string | null;
  hwid: string | null;
}

// Camera info from backend OpenCV detection (ground truth indices)
export interface CameraInfo {
  opencvIndex: number; // OpenCV camera index (from backend, ground truth)
  label: string;       // Camera name from system_profiler or fallback
}

// Camera selection in wizard
export interface CameraSelection {
  opencvIndex: number; // OpenCV camera index (key, ground truth from backend)
  label: string;       // display label
  name: string;        // "front_cam" | "hand_cam" | "side_cam"
  included: boolean;
}

// Recording configuration
export interface RecordingConfig {
  repoId: string;
  task: string;
  numEpisodes: number;
  episodeTimeS: number;
  resetTimeS: number;
  displayData: boolean;
  cameraFps: number;
  cameraWidth: number;
  cameraHeight: number;
}

// API start response
export interface StartResponse {
  process_id: string;
  message: string;
}

// Wizard state
export interface WizardState {
  currentStep: number; // 0-5
  completedSteps: boolean[];

  // Step 0: Robot Type
  robotMode: RobotMode | null;

  // Step 1: Ports
  detectedPorts: PortInfo[];
  portAssignments: Record<string, string>; // role → port path

  // Step 2: Cameras
  camerasStepVisited: boolean;
  detectedCameras: CameraInfo[];
  cameraSelections: CameraSelection[];

  // Step 3: Calibration
  calibrationFiles: Record<string, string[]>; // "robots/so101_follower" → filenames
  calibrationSelections: Record<string, string | null>; // role → filename or "new" or null
  newCalibrationNames: Record<string, string>; // role → calibration name (must follow {base}_left / {base}_right for bimanual)

  // Step 4: Teleoperation
  teleStepVisited: boolean;
  teleProcessId: string | null;

  // Step 5: Recording
  recordStepVisited: boolean;
  recordingConfig: RecordingConfig;
  recordProcessId: string | null;
}

// Port roles by mode
export const SINGLE_PORT_ROLES = ["follower", "leader"] as const;
export const BIMANUAL_PORT_ROLES = [
  "left_follower",
  "right_follower",
  "left_leader",
  "right_leader",
] as const;

// Camera name options
export const CAMERA_NAME_OPTIONS = [
  "front_cam",
  "hand_cam",
  "side_cam",
] as const;

// Calibration directory mapping
export function getCalibrationPaths(mode: RobotMode): { role: string; category: string; robotType: string }[] {
  if (mode === "single") {
    return [
      { role: "follower", category: "robots", robotType: "so101_follower" },
      { role: "leader", category: "teleoperators", robotType: "so101_leader" },
    ];
  }
  // Bimanual wrappers (bi_so101_follower / bi_so101_leader) create SO101Follower
  // and SO101Leader sub-arm instances internally, which look for calibration files
  // under so101_follower / so101_leader — NOT bi_so101_*.
  return [
    { role: "left_follower", category: "robots", robotType: "so101_follower" },
    { role: "right_follower", category: "robots", robotType: "so101_follower" },
    { role: "left_leader", category: "teleoperators", robotType: "so101_leader" },
    { role: "right_leader", category: "teleoperators", robotType: "so101_leader" },
  ];
}

// Step definitions
export const STEPS = [
  { label: "Robot Type", description: "Choose your robot arm configuration" },
  { label: "Ports", description: "Detect and assign USB device ports" },
  { label: "Cameras", description: "Select and name your cameras" },
  { label: "Calibration", description: "Choose calibration for each arm" },
  { label: "Teleoperate", description: "Test robot teleoperation" },
  { label: "Record", description: "Record training data" },
] as const;

// Initial state
export const INITIAL_RECORDING_CONFIG: RecordingConfig = {
  repoId: "",
  task: "",
  numEpisodes: 10,
  episodeTimeS: 60,
  resetTimeS: 10,
  displayData: true,
  cameraFps: 30,
  cameraWidth: 640,
  cameraHeight: 480,
};

export const INITIAL_STATE: WizardState = {
  currentStep: 0,
  completedSteps: [false, false, false, false, false, false],
  robotMode: null,
  detectedPorts: [],
  portAssignments: {},
  camerasStepVisited: false,
  detectedCameras: [],
  cameraSelections: [],
  calibrationFiles: {},
  calibrationSelections: {},
  newCalibrationNames: {},
  teleStepVisited: false,
  teleProcessId: null,
  recordStepVisited: false,
  recordingConfig: { ...INITIAL_RECORDING_CONFIG },
  recordProcessId: null,
};

// ─── Bimanual calibration naming validation ─────────────────────────────────

export interface BimanualValidationResult {
  valid: boolean;
  followerBaseId: string | null;
  leaderBaseId: string | null;
  errors: string[];
}

/**
 * Resolve the effective calibration name for a role.
 * Returns null if the role has no selection yet.
 */
function resolveCalName(
  role: string,
  selections: Record<string, string | null>,
  newNames: Record<string, string>,
): string | null {
  const sel = selections[role];
  if (sel === undefined || sel === null) return null;
  if (sel === "new") {
    const name = (newNames[role] || "").trim();
    return name || null;
  }
  return sel.replace(/\.json$/, "");
}

/**
 * Validate that bimanual left/right calibration names share a common prefix
 * and use the correct _left / _right suffixes.
 *
 * Returns early with valid=false and empty errors when selections are incomplete
 * (user hasn't filled everything yet — no premature error messages).
 */
export function validateBimanualCalibrationNames(
  selections: Record<string, string | null>,
  newNames: Record<string, string>,
): BimanualValidationResult {
  const result: BimanualValidationResult = {
    valid: false,
    followerBaseId: null,
    leaderBaseId: null,
    errors: [],
  };

  const pairs: Array<{
    label: string;
    leftRole: string;
    rightRole: string;
    setBase: (id: string) => void;
  }> = [
    {
      label: "Follower",
      leftRole: "left_follower",
      rightRole: "right_follower",
      setBase: (id) => { result.followerBaseId = id; },
    },
    {
      label: "Leader",
      leftRole: "left_leader",
      rightRole: "right_leader",
      setBase: (id) => { result.leaderBaseId = id; },
    },
  ];

  for (const pair of pairs) {
    const leftName = resolveCalName(pair.leftRole, selections, newNames);
    const rightName = resolveCalName(pair.rightRole, selections, newNames);

    // Not ready yet — no errors, just not valid
    if (!leftName || !rightName) return result;

    if (!leftName.endsWith("_left")) {
      result.errors.push(
        `Left ${pair.label} calibration name "${leftName}" must end with "_left" (e.g. "my_robot_left").`
      );
    }
    if (!rightName.endsWith("_right")) {
      result.errors.push(
        `Right ${pair.label} calibration name "${rightName}" must end with "_right" (e.g. "my_robot_right").`
      );
    }

    if (result.errors.length > 0) continue;

    const leftPrefix = leftName.slice(0, -"_left".length);
    const rightPrefix = rightName.slice(0, -"_right".length);

    if (leftPrefix !== rightPrefix) {
      result.errors.push(
        `${pair.label} calibration names must share the same base prefix — got "${leftPrefix}" (left) vs "${rightPrefix}" (right).`
      );
    } else {
      pair.setBase(leftPrefix);
    }
  }

  result.valid = result.errors.length === 0 && result.followerBaseId !== null && result.leaderBaseId !== null;
  return result;
}
