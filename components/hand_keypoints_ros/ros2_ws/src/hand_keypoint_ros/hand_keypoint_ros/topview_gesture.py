"""VIPLab landmark hand-shape classifier.

The official MediaPipe Gesture Recognizer remains the source of the 21 hand
landmarks.  This module replaces only its canned gesture head with a small,
deterministic classifier designed for the fixed CAM4 top view.  The optional
``right_ee`` profile is deliberately narrower than a second generic
classifier. It accepts a fist only when at least three non-thumb *3-D
anatomical PIP* joints are compact, while the image projection is not an
Open-Palm consensus. This is specific to the right end-effector camera,
where perspective and self-occlusion make a true fist look ambiguous in 2-D
but can also make a partially curled hand look falsely closed.

The public labels intentionally stay compatible with the existing ROS and
overlay contract: ``Open_Palm``, ``Closed_Fist``, and the fail-closed
rejection category ``None``.
"""

import hashlib

import numpy as np


GESTURE_NAMES = ('Closed_Fist', 'Open_Palm')
OUTPUT_NAMES = ('None',) + GESTURE_NAMES
CLASSIFIER_NAME = 'VIPLab Top-View Landmark Gesture Classifier'
CLASSIFIER_VERSION = 'landmark-geometry-v2-world-closed'
GESTURE_PROFILES = ('topview', 'right_ee')
RIGHT_EE_CLASSIFIER_NAME = 'VIPLab Right-EE Landmark Gesture Classifier'
RIGHT_EE_CLASSIFIER_VERSION = (
    'landmark-geometry-v3-right-ee-world-pip-compact')

# These thresholds use the angle at each non-thumb PIP joint, measured from
# MCP -> PIP -> fingertip in pixel coordinates.  The preceding live CAM4
# experiment used the same 145-degree open criterion and showed that it was
# invariant across the full image-plane rotation where the canned classifier
# failed.  Three-of-four consensus tolerates one occluded glove finger.
OPEN_ANGLE_DEG = 145.0
OPEN_DIP_ANGLE_DEG = 135.0
CLOSED_ANGLE_DEG = 115.0
OPEN_STRAIGHTNESS = 0.78
CLOSED_STRAIGHTNESS = 0.72
WORLD_CLOSED_PIP_ANGLE_DEG = 125.0
WORLD_CLOSED_DIP_ANGLE_DEG = 145.0
WORLD_CLOSED_STRAIGHTNESS = 0.78
WORLD_STRONG_CLOSED_PIP_ANGLE_DEG = 110.0
WORLD_STRONG_CLOSED_DIP_ANGLE_DEG = 125.0
WORLD_STRONG_CLOSED_STRAIGHTNESS = 0.80
WORLD_STRONG_OPEN_PIP_ANGLE_DEG = 155.0
WORLD_STRONG_OPEN_DIP_ANGLE_DEG = 145.0
WORLD_STRONG_OPEN_STRAIGHTNESS = 0.85
# User-reviewed right-EE frames E/F/G have >=3 compact PIPs (the
# third-smallest is <=110 deg); the intermediate H pose is about 129 deg.
RIGHT_EE_WORLD_COMPACT_PIP_ANGLE_DEG = 110.0
MIN_CONSENSUS_FINGERS = 3
MIN_PALM_SCALE_PX = 18.0
MIN_HAND_DIAGONAL_PX = 60.0
FINGER_CHAINS = (
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

RULE_SPEC = (
    f'version={CLASSIFIER_VERSION};'
    f'open_angle_deg={OPEN_ANGLE_DEG:.1f};'
    f'open_dip_angle_deg={OPEN_DIP_ANGLE_DEG:.1f};'
    f'closed_angle_deg={CLOSED_ANGLE_DEG:.1f};'
    f'open_straightness={OPEN_STRAIGHTNESS:.2f};'
    f'closed_straightness={CLOSED_STRAIGHTNESS:.2f};'
    f'world_closed_pip_angle_deg={WORLD_CLOSED_PIP_ANGLE_DEG:.1f};'
    f'world_closed_dip_angle_deg={WORLD_CLOSED_DIP_ANGLE_DEG:.1f};'
    f'world_closed_straightness={WORLD_CLOSED_STRAIGHTNESS:.2f};'
    f'world_strong_closed_pip_angle_deg={WORLD_STRONG_CLOSED_PIP_ANGLE_DEG:.1f};'
    f'world_strong_closed_dip_angle_deg={WORLD_STRONG_CLOSED_DIP_ANGLE_DEG:.1f};'
    f'world_strong_closed_straightness={WORLD_STRONG_CLOSED_STRAIGHTNESS:.2f};'
    f'world_strong_open_pip_angle_deg={WORLD_STRONG_OPEN_PIP_ANGLE_DEG:.1f};'
    f'world_strong_open_dip_angle_deg={WORLD_STRONG_OPEN_DIP_ANGLE_DEG:.1f};'
    f'world_strong_open_straightness={WORLD_STRONG_OPEN_STRAIGHTNESS:.2f};'
    f'min_consensus_fingers={MIN_CONSENSUS_FINGERS};'
    f'min_palm_scale_px={MIN_PALM_SCALE_PX:.1f};'
    f'min_hand_diagonal_px={MIN_HAND_DIAGONAL_PX:.1f};'
    'fingers=index,middle,ring,pinky;'
    'image_angle=mcp-pip-tip;'
    'world_pip_angle=mcp-pip-dip;world_dip_angle=pip-dip-tip;'
    'handedness=unused')
CLASSIFIER_SHA256 = hashlib.sha256(RULE_SPEC.encode('utf-8')).hexdigest()
RIGHT_EE_RULE_SPEC = (
    f'base={CLASSIFIER_VERSION};'
    'profile=right_ee;'
    f'world_compact_pip_angle_deg={RIGHT_EE_WORLD_COMPACT_PIP_ANGLE_DEG:.1f};'
    'world_compact_requires_fingers>=3;'
    'image_closed_requires_extended_fingers<3;'
    'scope=right_ee_only')
RIGHT_EE_CLASSIFIER_SHA256 = hashlib.sha256(
    RIGHT_EE_RULE_SPEC.encode('utf-8')).hexdigest()


def classifier_metadata(profile='topview'):
    """Return provenance for the selected public classification profile."""
    if profile == 'topview':
        return {
            'name': CLASSIFIER_NAME,
            'version': CLASSIFIER_VERSION,
            'sha256': CLASSIFIER_SHA256,
            'supported_gestures': GESTURE_NAMES,
        }
    if profile == 'right_ee':
        return {
            'name': RIGHT_EE_CLASSIFIER_NAME,
            'version': RIGHT_EE_CLASSIFIER_VERSION,
            'sha256': RIGHT_EE_CLASSIFIER_SHA256,
            'supported_gestures': GESTURE_NAMES,
        }
    raise ValueError(
        f'gesture profile must be one of {GESTURE_PROFILES}; got {profile!r}')


def _invalid_geometry(reason, classifier_version=CLASSIFIER_VERSION):
    return {
        'has_gesture': True,
        'category_name': 'None',
        'score': 0.55,
        'classifier': classifier_version,
        'quality_valid': False,
        'rejection_reason': str(reason),
        'finger_angles_deg': [],
        'dip_angles_deg': [],
        'finger_straightness': [],
        'extended_fingers': 0,
        'curled_fingers': 0,
        'world_geometry_valid': False,
        'world_pip_angles_deg': [],
        'world_dip_angles_deg': [],
        'world_finger_straightness': [],
        'world_curled_fingers': 0,
        'world_strongly_curled_fingers': 0,
        'world_strongly_extended_fingers': 0,
        'world_compact_pip_fingers': 0,
        'decision_path': 'invalid',
    }


def _joint_angle_deg(points_xy, first, pivot, last):
    first_vector = points_xy[first] - points_xy[pivot]
    last_vector = points_xy[last] - points_xy[pivot]
    denominator = (
        float(np.linalg.norm(first_vector))
        * float(np.linalg.norm(last_vector))
    )
    if denominator <= 1e-6:
        return None
    cosine = float(np.dot(first_vector, last_vector)) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _world_finger_features(points_world):
    """Return anatomical 3-D finger features, or ``None`` when unusable."""
    if points_world is None:
        return None
    points = np.asarray(points_world, dtype=np.float32)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        return None

    pip_angles = []
    dip_angles = []
    straightness = []
    for mcp, pip, dip, tip in FINGER_CHAINS:
        # Unlike the historical image-plane composite angle, these are the
        # anatomical PIP and DIP joints. This preserves flexion that projects
        # mostly onto camera Z when the back of a fist faces CAM4.
        pip_angles.append(_joint_angle_deg(points, mcp, pip, dip))
        dip_angles.append(_joint_angle_deg(points, pip, dip, tip))
        chain_length = sum(
            float(np.linalg.norm(points[end] - points[start]))
            for start, end in ((mcp, pip), (pip, dip), (dip, tip))
        )
        if chain_length <= 1e-6:
            straightness.append(None)
        else:
            straightness.append(
                float(np.linalg.norm(points[tip] - points[mcp]))
                / chain_length
            )
    if any(
        value is None
        for values in (pip_angles, dip_angles, straightness)
        for value in values
    ):
        return None
    return pip_angles, dip_angles, straightness


def classify(points_xy, points_world=None, profile='topview'):
    """Classify 21 MediaPipe landmarks supplied in image pixel coordinates.

    Angles are unchanged by translation, uniform scale, image-plane rotation,
    and mirroring.  Handedness is therefore deliberately not an input, making
    the decision path identical for the surgeon's left and right hands.

    ``points_world`` is MediaPipe's optional hand-centred 21x3 world skeleton.
    It is used only as a Closed-Fist auxiliary cue; no camera-coordinate claim
    is made from it. ``score`` is a deterministic decision-margin indicator,
    not a calibrated
    probability.  ``profile='topview'`` is the default and preserves the
    existing CAM4 decision rules. ``profile='right_ee'`` instead requires a
    measured 3-D PIP compactness consensus for a closed fist; it does not
    promote a perspective-distorted 2-D partial curl by itself. Invalid,
    collapsed, and ambiguous hand geometry all returns the explicit
    fail-closed ``None`` class.
    """
    metadata = classifier_metadata(profile)
    points = np.asarray(points_xy, dtype=np.float32)
    if points.shape != (21, 2) or not np.all(np.isfinite(points)):
        return _invalid_geometry(
            'invalid_landmark_array', metadata['version'])

    palm_scale = float(np.median([
        np.linalg.norm(points[index] - points[0])
        for index in (5, 9, 13, 17)
    ]))
    bbox_size = np.ptp(points, axis=0)
    hand_diagonal = float(np.linalg.norm(bbox_size))
    if (
        palm_scale < MIN_PALM_SCALE_PX
        or hand_diagonal < MIN_HAND_DIAGONAL_PX
    ):
        return _invalid_geometry(
            'hand_geometry_too_small', metadata['version'])

    pip_angles = []
    dip_angles = []
    straightness = []
    for mcp, pip, dip, tip in FINGER_CHAINS:
        pip_angles.append(_joint_angle_deg(points, mcp, pip, tip))
        dip_angles.append(_joint_angle_deg(points, pip, dip, tip))
        chain_length = sum(
            float(np.linalg.norm(points[end] - points[start]))
            for start, end in ((mcp, pip), (pip, dip), (dip, tip))
        )
        if chain_length <= 1e-6:
            straightness.append(None)
        else:
            straightness.append(
                float(np.linalg.norm(points[tip] - points[mcp]))
                / chain_length
            )
    if any(
        value is None
        for values in (pip_angles, dip_angles, straightness)
        for value in values
    ):
        return _invalid_geometry(
            'degenerate_finger_geometry', metadata['version'])

    extended = sum(
        pip_angle >= OPEN_ANGLE_DEG
        and dip_angle >= OPEN_DIP_ANGLE_DEG
        and ratio >= OPEN_STRAIGHTNESS
        for pip_angle, dip_angle, ratio in zip(
            pip_angles, dip_angles, straightness)
    )
    curled = sum(
        pip_angle <= CLOSED_ANGLE_DEG
        and ratio <= CLOSED_STRAIGHTNESS
        for pip_angle, ratio in zip(pip_angles, straightness)
    )
    world_features = _world_finger_features(points_world)
    if world_features is None:
        world_pip_angles = []
        world_dip_angles = []
        world_straightness = []
        world_curled = 0
        world_strongly_curled = 0
        world_strongly_extended = 0
        world_compact_pip = 0
    else:
        (world_pip_angles, world_dip_angles,
         world_straightness) = world_features
        world_curled = sum(
            pip_angle <= WORLD_CLOSED_PIP_ANGLE_DEG
            and (
                dip_angle <= WORLD_CLOSED_DIP_ANGLE_DEG
                or ratio <= WORLD_CLOSED_STRAIGHTNESS
            )
            for pip_angle, dip_angle, ratio in zip(
                world_pip_angles, world_dip_angles, world_straightness)
        )
        world_strongly_curled = sum(
            pip_angle <= WORLD_STRONG_CLOSED_PIP_ANGLE_DEG
            and dip_angle <= WORLD_STRONG_CLOSED_DIP_ANGLE_DEG
            and ratio <= WORLD_STRONG_CLOSED_STRAIGHTNESS
            for pip_angle, dip_angle, ratio in zip(
                world_pip_angles, world_dip_angles, world_straightness)
        )
        world_strongly_extended = sum(
            pip_angle >= WORLD_STRONG_OPEN_PIP_ANGLE_DEG
            and dip_angle >= WORLD_STRONG_OPEN_DIP_ANGLE_DEG
            and ratio >= WORLD_STRONG_OPEN_STRAIGHTNESS
            for pip_angle, dip_angle, ratio in zip(
                world_pip_angles, world_dip_angles, world_straightness)
        )
        world_compact_pip = sum(
            pip_angle <= RIGHT_EE_WORLD_COMPACT_PIP_ANGLE_DEG
            for pip_angle in world_pip_angles
        )
    open_consensus_angle = sorted(pip_angles, reverse=True)[
        MIN_CONSENSUS_FINGERS - 1]
    closed_consensus_angle = sorted(pip_angles)[
        MIN_CONSENSUS_FINGERS - 1]

    # A unanimously *strong* 3-D curl is a safety veto for a deceptively
    # straight 2-D projection. This is the palm-down/back-of-hand fist failure
    # mode seen in CAM4. Three-of-four world consensus is accepted only when
    # the image path is not already a strong Open-Palm consensus.
    strong_unanimous_world_closed = (
        world_features is not None
        and world_strongly_curled == len(FINGER_CHAINS)
        and world_strongly_extended == 0
    )
    consensus_world_closed = (
        world_features is not None
        and world_curled >= MIN_CONSENSUS_FINGERS
    )
    classifiable_world_closed = (
        consensus_world_closed and world_strongly_extended == 0
    )
    right_ee_compact_world_closed = (
        profile == 'right_ee'
        and world_features is not None
        and world_compact_pip >= MIN_CONSENSUS_FINGERS
        # Never convert an image-plane open consensus into a fist merely
        # because the monocular 3-D reconstruction happens to be compact.
        and extended < MIN_CONSENSUS_FINGERS
    )

    world_image_open_conflict = (
        extended >= MIN_CONSENSUS_FINGERS
        and consensus_world_closed
        and not strong_unanimous_world_closed
    )

    if strong_unanimous_world_closed:
        world_consensus_angle = sorted(world_pip_angles)[
            MIN_CONSENSUS_FINGERS - 1]
        margin = np.clip(
            (WORLD_CLOSED_PIP_ANGLE_DEG - world_consensus_angle) / 65.0,
            0.0,
            1.0,
        )
        category = 'Closed_Fist'
        score = 0.60 + 0.39 * float(margin)
        decision_path = 'world_closed_strong_unanimous'
    elif right_ee_compact_world_closed:
        world_compact_angle = sorted(world_pip_angles)[
            MIN_CONSENSUS_FINGERS - 1]
        margin = np.clip(
            (RIGHT_EE_WORLD_COMPACT_PIP_ANGLE_DEG - world_compact_angle)
            / 50.0,
            0.0,
            1.0,
        )
        category = 'Closed_Fist'
        score = 0.60 + 0.39 * float(margin)
        decision_path = 'right_ee_world_pip_compact'
    elif world_image_open_conflict:
        # Do not turn a disagreement between the monocular 2-D and 3-D heads
        # into either an Open intent or a fist. Live calibration can later
        # narrow this fail-closed rejection region.
        category = 'None'
        score = 0.65
        decision_path = 'world_image_conflict'
    elif extended >= MIN_CONSENSUS_FINGERS:
        margin = np.clip(
            (open_consensus_angle - OPEN_ANGLE_DEG) / 30.0, 0.0, 1.0)
        category = 'Open_Palm'
        score = 0.55 + 0.44 * float(margin)
        decision_path = 'image_open'
    elif (
        profile == 'topview'
        and curled >= MIN_CONSENSUS_FINGERS
        and extended == 0
    ):
        margin = np.clip(
            (CLOSED_ANGLE_DEG - closed_consensus_angle) / 65.0,
            0.0,
            1.0,
        )
        category = 'Closed_Fist'
        score = 0.55 + 0.44 * float(margin)
        decision_path = 'image_closed'
    elif profile == 'topview' and classifiable_world_closed:
        world_consensus_angle = sorted(world_pip_angles)[
            MIN_CONSENSUS_FINGERS - 1]
        margin = np.clip(
            (WORLD_CLOSED_PIP_ANGLE_DEG - world_consensus_angle) / 65.0,
            0.0,
            1.0,
        )
        category = 'Closed_Fist'
        score = 0.58 + 0.40 * float(margin)
        decision_path = 'world_closed_consensus'
    else:
        # Ambiguous, partially open, and one/two-finger shapes fail closed.
        open_gap = max(0.0, OPEN_ANGLE_DEG - open_consensus_angle)
        closed_gap = max(
            0.0, closed_consensus_angle - CLOSED_ANGLE_DEG)
        margin = np.clip(min(open_gap, closed_gap) / 30.0, 0.0, 1.0)
        category = 'None'
        score = 0.55 + 0.34 * float(margin)
        decision_path = 'ambiguous'

    return {
        'has_gesture': True,
        'category_name': category,
        'score': float(score),
        'classifier': metadata['version'],
        'quality_valid': True,
        'rejection_reason': '' if category != 'None' else 'ambiguous_shape',
        'finger_angles_deg': [
            round(float(angle), 2) for angle in pip_angles],
        'dip_angles_deg': [
            round(float(angle), 2) for angle in dip_angles],
        'finger_straightness': [
            round(float(ratio), 4) for ratio in straightness],
        'extended_fingers': int(extended),
        'curled_fingers': int(curled),
        'world_geometry_valid': world_features is not None,
        'world_pip_angles_deg': [
            round(float(angle), 2) for angle in world_pip_angles],
        'world_dip_angles_deg': [
            round(float(angle), 2) for angle in world_dip_angles],
        'world_finger_straightness': [
            round(float(ratio), 4) for ratio in world_straightness],
        'world_curled_fingers': int(world_curled),
        'world_strongly_curled_fingers': int(world_strongly_curled),
        'world_strongly_extended_fingers': int(world_strongly_extended),
        'world_compact_pip_fingers': int(world_compact_pip),
        'decision_path': decision_path,
    }
