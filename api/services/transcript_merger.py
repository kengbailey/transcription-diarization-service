"""Transcript merger - aligns Whisper transcription with diarization segments."""

import logging
from typing import Optional


logger = logging.getLogger(__name__)


class TranscriptMerger:
    """Merges Whisper transcription with speaker diarization results."""
    
    def merge_transcription_with_diarization(
        self,
        whisper_result: dict,
        diarization_result: dict,
        speaker_mapping: Optional[dict] = None,
        speaker_confidences: Optional[dict] = None
    ) -> dict:
        """Merge Whisper transcription with speaker diarization.
        
        Aligns each word from Whisper to the speaker from diarization
        based on timestamp overlap, then groups into speaker turns.
        
        Args:
            whisper_result: Whisper API response with 'text', 'segments', 'words'
            diarization_result: Diarization result with 'segments'
            speaker_mapping: Optional dict mapping speaker labels to names
            speaker_confidences: Optional dict with confidence scores per speaker
            
        Returns:
            Merged result with speaker-attributed segments
        """
        # Get words from Whisper (prefer word-level, fallback to segment-level)
        words = whisper_result.get("words", [])
        
        if not words:
            # Try to extract from segments
            for seg in whisper_result.get("segments", []):
                if "words" in seg:
                    words.extend(seg["words"])
        
        if not words:
            # Fall back to segment-level alignment
            return self._merge_segment_level(
                whisper_result, diarization_result, speaker_mapping, speaker_confidences
            )
        
        # Get diarization segments
        diar_segments = diarization_result.get("segments", [])
        
        if not diar_segments:
            # No diarization, return transcription as single speaker
            return {
                "text": whisper_result.get("text", ""),
                "segments": [{
                    "speaker": "SPEAKER_00",
                    "identified_as": speaker_mapping.get("SPEAKER_00") if speaker_mapping else None,
                    "confidence": speaker_confidences.get("SPEAKER_00") if speaker_confidences else None,
                    "start": 0,
                    "end": whisper_result.get("duration", 0),
                    "text": whisper_result.get("text", "")
                }],
                "words": words,
                "num_speakers": 1,
                "duration": whisper_result.get("duration", 0)
            }
        
        # Assign each word to a speaker
        word_assignments = self._assign_words_to_speakers(words, diar_segments)
        
        # Group consecutive words by speaker into turns
        speaker_turns = self._group_words_into_turns(word_assignments)
        
        # Apply speaker mapping if provided
        if speaker_mapping:
            for turn in speaker_turns:
                original_speaker = turn["speaker"]
                turn["identified_as"] = speaker_mapping.get(original_speaker)
                if speaker_confidences:
                    turn["confidence"] = speaker_confidences.get(original_speaker)
        
        # Build full text
        full_text = whisper_result.get("text", "")
        if not full_text:
            full_text = " ".join(w.get("word", "") for w in words)
        
        return {
            "text": full_text,
            "segments": speaker_turns,
            "num_speakers": diarization_result.get("num_speakers", len(set(t["speaker"] for t in speaker_turns))),
            "duration": whisper_result.get("duration", diarization_result.get("audio_duration", 0)),
            "language": whisper_result.get("language")
        }
    
    def _assign_words_to_speakers(
        self,
        words: list[dict],
        diar_segments: list[dict]
    ) -> list[dict]:
        """Assign each word to a speaker based on timestamp overlap.
        
        Args:
            words: List of words with 'start', 'end', 'word' keys
            diar_segments: List of diarization segments with 'start', 'end', 'speaker'
            
        Returns:
            List of words with added 'speaker' key
        """
        assigned = []
        
        for word in words:
            word_start = word.get("start", 0)
            word_end = word.get("end", word_start)
            word_mid = (word_start + word_end) / 2
            
            # Find the speaker for this word's midpoint
            speaker = self._find_speaker_at_time(word_mid, diar_segments)
            
            assigned.append({
                **word,
                "speaker": speaker or "SPEAKER_UNKNOWN"
            })
        
        return assigned
    
    def _find_speaker_at_time(
        self,
        timestamp: float,
        diar_segments: list[dict]
    ) -> Optional[str]:
        """Find which speaker was talking at a given timestamp.
        
        Args:
            timestamp: Time in seconds
            diar_segments: List of diarization segments
            
        Returns:
            Speaker label or None if no speaker found
        """
        for seg in diar_segments:
            if seg["start"] <= timestamp <= seg["end"]:
                return seg["speaker"]
        
        # If exact match not found, find nearest segment
        min_distance = float("inf")
        nearest_speaker = None
        
        for seg in diar_segments:
            # Distance to segment
            if timestamp < seg["start"]:
                distance = seg["start"] - timestamp
            elif timestamp > seg["end"]:
                distance = timestamp - seg["end"]
            else:
                distance = 0
            
            if distance < min_distance:
                min_distance = distance
                nearest_speaker = seg["speaker"]
        
        # Only assign if within 0.5 second of a segment
        if min_distance < 0.5:
            return nearest_speaker
        
        return None
    
    def _group_words_into_turns(
        self,
        words: list[dict]
    ) -> list[dict]:
        """Group consecutive words by speaker into turns.
        
        Args:
            words: List of words with 'speaker' assignments
            
        Returns:
            List of speaker turns with aggregated text
        """
        if not words:
            return []
        
        turns = []
        current_turn = None
        
        for word in words:
            speaker = word.get("speaker", "SPEAKER_UNKNOWN")
            word_text = word.get("word", "").strip()
            
            if not word_text:
                continue
            
            if current_turn is None or current_turn["speaker"] != speaker:
                # Start new turn
                if current_turn is not None:
                    turns.append(current_turn)
                
                current_turn = {
                    "speaker": speaker,
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "text": word_text,
                    "words": [word]
                }
            else:
                # Continue current turn
                current_turn["end"] = word.get("end", current_turn["end"])
                current_turn["text"] += " " + word_text
                current_turn["words"].append(word)
        
        # Add final turn
        if current_turn is not None:
            turns.append(current_turn)
        
        # Clean up turns (remove words list, add duration)
        for turn in turns:
            turn["duration"] = round(turn["end"] - turn["start"], 3)
            del turn["words"]  # Remove word-level detail from segment
        
        return turns
    
    def _merge_segment_level(
        self,
        whisper_result: dict,
        diarization_result: dict,
        speaker_mapping: Optional[dict] = None,
        speaker_confidences: Optional[dict] = None
    ) -> dict:
        """Fallback: merge at segment level when word timestamps unavailable.
        
        Aligns Whisper segments with diarization segments based on overlap.
        """
        whisper_segments = whisper_result.get("segments", [])
        diar_segments = diarization_result.get("segments", [])
        
        merged_segments = []
        
        for wseg in whisper_segments:
            seg_start = wseg.get("start", 0)
            seg_end = wseg.get("end", 0)
            seg_mid = (seg_start + seg_end) / 2
            
            # Find speaker
            speaker = self._find_speaker_at_time(seg_mid, diar_segments) or "SPEAKER_UNKNOWN"
            
            merged_seg = {
                "speaker": speaker,
                "start": seg_start,
                "end": seg_end,
                "duration": round(seg_end - seg_start, 3),
                "text": wseg.get("text", "").strip()
            }
            
            if speaker_mapping:
                merged_seg["identified_as"] = speaker_mapping.get(speaker)
            if speaker_confidences:
                merged_seg["confidence"] = speaker_confidences.get(speaker)
            
            merged_segments.append(merged_seg)
        
        return {
            "text": whisper_result.get("text", ""),
            "segments": merged_segments,
            "num_speakers": diarization_result.get("num_speakers", 1),
            "duration": whisper_result.get("duration", diarization_result.get("audio_duration", 0)),
            "language": whisper_result.get("language")
        }
