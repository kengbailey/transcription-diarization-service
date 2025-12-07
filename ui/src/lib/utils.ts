import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`
  }
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}m ${secs}s`
}

// Speaker colors for visual distinction
export const SPEAKER_COLORS = [
  { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/50', solid: '#3b82f6' },
  { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/50', solid: '#10b981' },
  { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/50', solid: '#a855f7' },
  { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/50', solid: '#f97316' },
  { bg: 'bg-pink-500/20', text: 'text-pink-400', border: 'border-pink-500/50', solid: '#ec4899' },
  { bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/50', solid: '#06b6d4' },
  { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/50', solid: '#eab308' },
  { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/50', solid: '#ef4444' },
]

export function getSpeakerColor(index: number) {
  return SPEAKER_COLORS[index % SPEAKER_COLORS.length]
}
