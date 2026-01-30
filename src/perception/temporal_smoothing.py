"""
Temporal Smoothing for Ball Detection
Tracks detections across frames, interpolates gaps, and filters isolated detections.
"""
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from src.types import Detection, TrackedObject


class TemporalSmoother:
    """
    Temporal smoothing for ball detections.
    
    Features:
    - Tracks detections across frames
    - Interpolates gaps between detections (fills missing frames)
    - Filters isolated detections (removes single-frame detections)
    - Maintains smooth trajectories
    """
    
    def __init__(
        self,
        min_track_length: int = 2,  # Minimum frames for a valid track
        max_gap_fill: int = 5,  # Maximum frames to interpolate
        isolation_threshold: int = 1,  # Frames before considering isolated
        velocity_threshold: float = 100.0  # Max pixels/frame for valid motion
    ):
        """
        Initialize temporal smoother.
        
        Args:
            min_track_length: Minimum consecutive frames for valid track
            max_gap_fill: Maximum gap to interpolate (frames)
            isolation_threshold: Frames before considering detection isolated
            velocity_threshold: Maximum velocity for valid ball motion (pixels/frame)
        """
        self.min_track_length = min_track_length
        self.max_gap_fill = max_gap_fill
        self.isolation_threshold = isolation_threshold
        self.velocity_threshold = velocity_threshold
        
        # Track history: track_id -> List[(frame_id, detection, center_x, center_y)]
        self.track_history: Dict[int, List[Tuple[int, Detection, float, float]]] = defaultdict(list)
        
        # Active tracks: track_id -> last_frame_id
        self.active_tracks: Dict[int, int] = {}
        
        # Frame cache: frame_id -> List[Detection]
        self.frame_cache: Dict[int, List[Detection]] = {}
    
    def _get_center(self, detection: Detection) -> Tuple[float, float]:
        """Get center point of detection."""
        x, y, w, h = detection.bbox
        return (x + w / 2, y + h / 2)
    
    def _calculate_velocity(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        frame_diff: int
    ) -> float:
        """Calculate velocity between two points."""
        if frame_diff == 0:
            return 0.0
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance / frame_diff
    
    def _interpolate_detection(
        self,
        det1: Detection,
        det2: Detection,
        frame1: int,
        frame2: int,
        target_frame: int
    ) -> Detection:
        """
        Interpolate detection between two frames.
        
        Args:
            det1: Detection at frame1
            det2: Detection at frame2
            frame1: First frame number
            frame2: Second frame number
            target_frame: Frame to interpolate for
        
        Returns:
            Interpolated detection
        """
        # Calculate interpolation factor
        if frame2 == frame1:
            alpha = 0.5
        else:
            alpha = (target_frame - frame1) / (frame2 - frame1)
        
        # Interpolate bbox
        x1, y1, w1, h1 = det1.bbox
        x2, y2, w2, h2 = det2.bbox
        
        x = x1 + (x2 - x1) * alpha
        y = y1 + (y2 - y1) * alpha
        w = w1 + (w2 - w1) * alpha
        h = h1 + (h2 - h1) * alpha
        
        # Interpolate confidence (use minimum to be conservative)
        confidence = min(det1.confidence, det2.confidence) * 0.9  # Slightly lower for interpolated
        
        return Detection(
            class_id=det1.class_id,
            confidence=confidence,
            bbox=(x, y, w, h),
            class_name=det1.class_name
        )
    
    def update(
        self,
        detections: List[Detection],
        frame_id: int,
        tracked_objects: Optional[List[TrackedObject]] = None
    ) -> Tuple[List[Detection], Dict[int, List[int]]]:
        """
        Update smoother with new detections.
        
        Args:
            detections: Detections for current frame
            frame_id: Current frame number
            tracked_objects: Optional tracked objects (if using tracker)
        
        Returns:
            Tuple of (smoothed_detections, track_info)
            track_info: Dict mapping track_id -> list of frame_ids in track
        """
        # Store original detections
        self.frame_cache[frame_id] = detections.copy()
        
        # If we have tracked objects, use track IDs
        if tracked_objects:
            # Map detections to track IDs
            track_id_map = {}
            for obj in tracked_objects:
                if obj.detection.class_name == 'ball':
                    # Find matching detection
                    for det in detections:
                        if det.class_name == 'ball':
                            # Check if same detection (by position)
                            x1, y1, w1, h1 = det.bbox
                            x2, y2, w2, h2 = obj.detection.bbox
                            center1 = (x1 + w1/2, y1 + h1/2)
                            center2 = (x2 + w2/2, y2 + h2/2)
                            dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                            if dist < 20:  # Same detection
                                track_id_map[id(det)] = obj.object_id
                                break
            
            # Update track history
            for det in detections:
                if det.class_name == 'ball':
                    track_id = track_id_map.get(id(det), None)
                    if track_id is None:
                        # Create new track ID
                        track_id = len(self.track_history) + 1000
                    
                    center_x, center_y = self._get_center(det)
                    self.track_history[track_id].append((frame_id, det, center_x, center_y))
                    self.active_tracks[track_id] = frame_id
        else:
            # No tracking - use simple matching by position
            for det in detections:
                if det.class_name == 'ball':
                    # Try to match to existing track
                    center_x, center_y = self._get_center(det)
                    matched_track = None
                    
                    for track_id, history in self.track_history.items():
                        if not history:
                            continue
                        last_frame, last_det, last_x, last_y = history[-1]
                        
                        # Check if close to last position
                        dist = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                        frame_gap = frame_id - last_frame
                        
                        if frame_gap <= self.max_gap_fill and dist < self.velocity_threshold * frame_gap:
                            matched_track = track_id
                            break
                    
                    if matched_track is None:
                        # New track
                        matched_track = len(self.track_history) + 1000
                    
                    self.track_history[matched_track].append((frame_id, det, center_x, center_y))
                    self.active_tracks[matched_track] = frame_id
        
        # Clean up old tracks
        tracks_to_remove = []
        for track_id, last_frame in self.active_tracks.items():
            if frame_id - last_frame > self.max_gap_fill + 5:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
        
        # Generate smoothed detections for current frame
        smoothed_detections = self._generate_smoothed_detections(frame_id)
        
        # Build track info
        track_info = {}
        for track_id, history in self.track_history.items():
            if history:
                track_info[track_id] = [frame for frame, _, _, _ in history]
        
        return smoothed_detections, track_info
    
    def _generate_smoothed_detections(self, frame_id: int) -> List[Detection]:
        """
        Generate smoothed detections for a frame.
        Includes original detections + interpolated detections from tracks.
        """
        smoothed = []
        
        # Add original detections from this frame
        has_original = False
        if frame_id in self.frame_cache:
            for det in self.frame_cache[frame_id]:
                if det.class_name == 'ball':
                    smoothed.append(det)
                    has_original = True
        
        # Check all tracks for interpolation opportunities
        for track_id, history in self.track_history.items():
            if not history:
                continue
            
            # Find detections before and after this frame
            before = None
            after = None
            
            for i, (f, det, x, y) in enumerate(history):
                if f == frame_id:
                    # Already have detection for this frame
                    before = (f, det, x, y, i)  # Use this as "before"
                    break
                elif f < frame_id:
                    before = (f, det, x, y, i)
                elif f > frame_id:
                    after = (f, det, x, y, i)
                    if before:  # We have both, can break
                        break
            
            # If we have both before and after, interpolate
            if before and after:
                frame1, det1, x1, y1, idx1 = before
                frame2, det2, x2, y2, idx2 = after
                
                # Check if gap is reasonable
                gap = frame2 - frame1
                if gap <= self.max_gap_fill and frame1 < frame_id < frame2:
                    # Check velocity
                    velocity = self._calculate_velocity(x1, y1, x2, y2, gap)
                    if velocity <= self.velocity_threshold:
                        # Interpolate
                        interp_det = self._interpolate_detection(det1, det2, frame1, frame2, frame_id)
                        smoothed.append(interp_det)
            # If we only have before and it's recent, extend the track
            elif before and not has_original:
                frame1, det1, x1, y1, idx1 = before
                gap = frame_id - frame1
                if gap <= self.max_gap_fill and gap > 0:
                    # Extend track forward (simple extrapolation)
                    # Use last known position with slight movement prediction
                    if len(history) >= 2:
                        # Use velocity from last two points
                        prev_frame, prev_det, prev_x, prev_y = history[-2]
                        velocity_x = (x1 - prev_x) / (frame1 - prev_frame) if (frame1 - prev_frame) > 0 else 0
                        velocity_y = (y1 - prev_y) / (frame1 - prev_frame) if (frame1 - prev_frame) > 0 else 0
                        
                        # Predict position
                        pred_x = x1 + velocity_x * gap
                        pred_y = y1 + velocity_y * gap
                        
                        # Check if prediction is reasonable
                        dist = np.sqrt((pred_x - x1)**2 + (pred_y - y1)**2)
                        if dist <= self.velocity_threshold * gap:
                            x, y, w, h = det1.bbox
                            extrap_det = Detection(
                                class_id=det1.class_id,
                                confidence=det1.confidence * 0.8,  # Lower confidence for extrapolated
                                bbox=(pred_x - w/2, pred_y - h/2, w, h),
                                class_name=det1.class_name
                            )
                            smoothed.append(extrap_det)
        
        # Remove duplicates (same position)
        unique_detections = []
        seen_positions = set()
        for det in smoothed:
            x, y, w, h = det.bbox
            center = (int(x + w/2), int(y + h/2))
            if center not in seen_positions:
                seen_positions.add(center)
                unique_detections.append(det)
        
        return unique_detections
    
    def get_all_smoothed_detections(self, start_frame: int, end_frame: int) -> Dict[int, List[Detection]]:
        """
        Get smoothed detections for all frames in range.
        This fills in gaps and provides complete detection history.
        
        Args:
            start_frame: First frame
            end_frame: Last frame (inclusive)
        
        Returns:
            Dict mapping frame_id -> List[Detection]
        """
        all_detections = {}
        
        for frame_id in range(start_frame, end_frame + 1):
            all_detections[frame_id] = self._generate_smoothed_detections(frame_id)
        
        return all_detections
    
    def reset(self):
        """Reset smoother state."""
        self.track_history.clear()
        self.active_tracks.clear()
        self.frame_cache.clear()
