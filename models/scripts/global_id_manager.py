import numpy as np
from collections import defaultdict, deque

def normalize(x):
    return x / (np.linalg.norm(x) + 1e-9)

def cosine(a, b):
    return np.dot(a, b) / (
        (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    )

class GlobalIDManager:

    def __init__(
        self,
        appearance_weight=0.30,
        temporal_weight=0.70,
        assignment_threshold=0.60,
        history_size=30,
        confirm_hits=3,
        expected_transition_frames=110,
        transition_sigma=40,
        max_transition_frames=160,
        lost_timeout=30,
        memory_timeout=900,
        short_occlusion_frames=60,
        short_occlusion_threshold=0.55
    ):

        self.appearance_weight = appearance_weight
        self.temporal_weight = temporal_weight

        self.assignment_threshold = assignment_threshold

        self.history_size = history_size
        self.confirm_hits = confirm_hits

        self.expected_transition_frames = expected_transition_frames
        self.transition_sigma = transition_sigma
        self.max_transition_frames = max_transition_frames

        self.lost_timeout = lost_timeout
        self.memory_timeout = memory_timeout

        self.short_occlusion_frames = short_occlusion_frames
        self.short_occlusion_threshold = short_occlusion_threshold

        self.next_gid = 0

        # Active GIDs definition
        self.active = {}
        self.current_assignments = {}

        # Creation of memory embedding for each GID
        self.memory = {}

        # Saving Information on last GID
        self.last_seen = {}
        self.last_camera = {}

        # Lost GIDs
        self.lost_frame = {}

        # Feature embedding history
        self.history = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

        # Lost IDs
        self.lost = set()
        self.reserved_gids = set()

        # Pending tracks for GIDs not yet confirmed
        self.pending_tracks = {}

    def add_observation(
        self,
        frame_id,
        cam_id,
        track_id,
        embedding,
        bbox=None,
        centroid=None
    ):

        emb = normalize(embedding)

        key = (cam_id, track_id)

        # Assigned GIDs
        if key in self.current_assignments:

            gid = self.current_assignments[key]

            self._update_gid(
                gid,
                emb,
                frame_id,
                cam_id
            )

            return gid
        
        # Pending GIDs
        if key in self.pending_tracks:
            return self._update_pending(
                key,
                emb,
                frame_id,
                cam_id
            )

        # Trying self camera reidentification
        candidate_gid, score = (
            self._find_active_same_camera_candidate(
                emb,
                frame_id,
                cam_id,
                track_id
            )
        )

        if candidate_gid is not None:

            self._assign_existing_gid(
                candidate_gid,
                cam_id,
                track_id,
                emb,
                frame_id
            )

            return candidate_gid


        # Trying lost GIDs reidentification
        candidate_gid, score = self._find_best_lost_candidate(
            emb,
            frame_id,
            cam_id
        )

        if (
            candidate_gid is not None
            and score >= self.assignment_threshold
        ):

            self.pending_tracks[key] = {
                "candidate_gid": candidate_gid,
                "hits": 1,
                "scores": [score]
            }

            self.reserved_gids.add(
                candidate_gid
            )

            return None

        return self._create_gid(
            emb,
            frame_id,
            cam_id,
            track_id
        )

    # Update to pending GID
    def _update_pending(
        self,
        key,
        emb,
        frame_id,
        cam_id
    ):

        state = self.pending_tracks[key]

        gid = state["candidate_gid"]

        lost_frame = self.lost_frame.get(
            gid,
            self.last_seen[gid]
        )

        dt = frame_id - lost_frame

        same_camera = (
            self.last_camera[gid] == cam_id
        )

        if (
            same_camera and
            dt <= self.short_occlusion_frames
        ):

            score = self._compute_appearance_score(
                gid,
                emb
            )

        else:

            score = self._compute_combined_score(
                gid,
                emb,
                frame_id
            )

        if score < self.assignment_threshold:

            self.reserved_gids.discard(gid)

            del self.pending_tracks[key]

            return self._create_gid(
                emb,
                frame_id,
                cam_id,
                key[1]
            )

        state["hits"] += 1
        state["scores"].append(score)

        if state["hits"] < self.confirm_hits:
            return None

        recent = state["scores"][-self.confirm_hits:]

        if np.mean(recent) < self.assignment_threshold:

            self.reserved_gids.discard(gid)

            del self.pending_tracks[key]

            return self._create_gid(
                emb,
                frame_id,
                cam_id,
                key[1]
            )

        self.reserved_gids.discard(gid)

        del self.pending_tracks[key]

        self._assign_existing_gid(
            gid,
            key[0],
            key[1],
            emb,
            frame_id
        )

        return gid

    # Search for best lost candidate
    def _find_best_lost_candidate(
        self,
        emb,
        frame_id,
        cam_id
    ):

        # Featured based same camera reidentification
        best_gid = None
        best_score = -1.0

        for gid in self.lost:

            if gid in self.reserved_gids:
                continue

            # misma cámara
            if self.last_camera[gid] != cam_id:
                continue

            lost_frame = self.lost_frame.get(
                gid,
                self.last_seen[gid]
            )

            dt = frame_id - lost_frame

            if dt > self.short_occlusion_frames:
                continue

            score = self._compute_appearance_score(
                gid,
                emb
            )

            if (
                score >= self.short_occlusion_threshold
                and score > best_score
            ):

                best_score = score
                best_gid = gid

        if best_gid is not None:
            return best_gid, best_score

        # temporal based cross camera reidentification
        best_gid = None
        best_score = -1.0

        for gid in self.lost:

            if gid in self.reserved_gids:
                continue

            # distinta cámara
            if self.last_camera[gid] == cam_id:
                continue

            lost_frame = self.lost_frame.get(
                gid,
                self.last_seen[gid]
            )

            dt = frame_id - lost_frame

            if dt > self.max_transition_frames:
                continue

            score = self._compute_combined_score(
                gid,
                emb,
                frame_id
            )

            if score > best_score:

                best_score = score
                best_gid = gid

        return best_gid, best_score

    # Evaluation of appearance score
    def _compute_appearance_score(
        self,
        gid,
        emb
    ):

        prototype = self._compute_prototype(
            gid
        )

        return float(
            cosine(
                emb,
                prototype
            )
        )

    def _find_active_same_camera_candidate(
        self,
        emb,
        frame_id,
        cam_id,
        current_track_id
    ):

        best_gid = None
        best_score = -1.0

        for gid, (active_cam, active_track) in self.active.items():

            if active_cam != cam_id:
                continue

            # mismo track -> ignorar
            if active_track == current_track_id:
                continue

            # Seen too quickly -> ignore
            if frame_id - self.last_seen[gid] < 5:
                continue

            score = self._compute_appearance_score(
                gid,
                emb
            )

            if (
                score >= self.short_occlusion_threshold
                and score > best_score
            ):

                best_score = score
                best_gid = gid

        return best_gid, best_score

    def _compute_combined_score(
        self,
        gid,
        emb,
        frame_id
    ):

        prototype = self._compute_prototype(gid)

        appearance_score = cosine(
            emb,
            prototype
        )

        lost_frame = self.lost_frame.get(
            gid,
            self.last_seen[gid]
        )

        dt = frame_id - lost_frame

        temporal_score = np.exp(
            -(
                (dt - self.expected_transition_frames) ** 2
            ) /
            (
                2.0 *
                (self.transition_sigma ** 2)
            )
        )

        return float(
            self.appearance_weight *
            appearance_score
            +
            self.temporal_weight *
            temporal_score
        )

    # Compute prototype embedding 
    def _compute_prototype(
        self,
        gid
    ):

        hist = list(
            self.history[gid]
        )

        if len(hist) == 0:
            return self.memory[gid]

        weights = np.linspace(
            0.5,
            1.0,
            len(hist)
        )

        weights /= (
            weights.sum() + 1e-9
        )

        proto = np.sum(
            [
                w * e
                for w, e in zip(
                    weights,
                    hist
                )
            ],
            axis=0
        )

        return normalize(proto)

    # Update GID information
    def _update_gid(
        self,
        gid,
        emb,
        frame_id,
        cam_id
    ):

        self.history[gid].append(
            emb
        )

        proto = self._compute_prototype(
            gid
        )

        self.memory[gid] = proto

        self.last_seen[gid] = frame_id

        self.last_camera[gid] = cam_id

    # Assign existing GID to a new track
    def _assign_existing_gid(
        self,
        gid,
        cam_id,
        track_id,
        emb,
        frame_id
    ):

        old_key = self.active.get(gid)

        if old_key is not None:

            self.current_assignments.pop(
                old_key,
                None
            )

        new_key = (
            cam_id,
            track_id
        )

        self.active[gid] = new_key

        self.current_assignments[
            new_key
        ] = gid

        self.lost.discard(gid)

        self.lost_frame.pop(
            gid,
            None
        )

        self._update_gid(
            gid,
            emb,
            frame_id,
            cam_id
        )

    # Create new GID
    def _create_gid(
        self,
        emb,
        frame_id,
        cam_id,
        track_id
    ):

        gid = self.next_gid

        self.next_gid += 1

        self.memory[gid] = emb

        self.last_seen[gid] = frame_id

        self.last_camera[gid] = cam_id

        self.history[gid].append(
            emb
        )

        self.active[gid] = (
            cam_id,
            track_id
        )

        self.current_assignments[
            (cam_id, track_id)
        ] = gid

        return gid

    # Erase GID information after timeout
    def cleanup(
        self,
        frame_id
    ):
        print(
            f"[STATUS] "
            f"ACTIVE={len(self.active)}"
            f"LOST={len(self.lost)}"
            f"PENDING={len(self.pending_tracks)}"
        )

        to_lost = []

        for gid, key in list(
            self.active.items()
        ):

            if (
                frame_id -
                self.last_seen[gid]
            ) > self.lost_timeout:

                to_lost.append(
                    (gid, key)
                )

        for gid, key in to_lost:

            self.active.pop(
                gid,
                None
            )

            self.current_assignments.pop(
                key,
                None
            )

            self.lost.add(gid)

            self.lost_frame[gid] = frame_id

        to_delete = []

        for gid in list(
            self.lost
        ):

            if (
                frame_id -
                self.lost_frame.get(
                    gid,
                    self.last_seen[gid]
                )
            ) > self.memory_timeout:

                to_delete.append(gid)

        for gid in to_delete:

            self.lost.discard(
                gid
            )

            self.reserved_gids.discard(
                gid
            )

            self.memory.pop(
                gid,
                None
            )

            self.last_seen.pop(
                gid,
                None
            )

            self.last_camera.pop(
                gid,
                None
            )

            self.lost_frame.pop(
                gid,
                None
            )

            self.history.pop(
                gid,
                None
            )