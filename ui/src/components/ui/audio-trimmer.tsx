import * as React from "react"
import WaveSurfer from "wavesurfer.js"
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.js"
import { Play, Pause, Scissors, RotateCcw } from "lucide-react"
import { Button } from "./button"
import { cn } from "@/lib/utils"

const MAX_DURATION = 30 // Maximum duration in seconds

interface AudioTrimmerProps {
  file: File
  onTrimmedAudio: (file: File) => void
  className?: string
}

export function AudioTrimmer({ file, onTrimmedAudio, className }: AudioTrimmerProps) {
  const containerRef = React.useRef<HTMLDivElement>(null)
  const wavesurferRef = React.useRef<WaveSurfer | null>(null)
  const regionsRef = React.useRef<RegionsPlugin | null>(null)
  const [isPlaying, setIsPlaying] = React.useState(false)
  const [isReady, setIsReady] = React.useState(false)
  const [duration, setDuration] = React.useState(0)
  const [regionStart, setRegionStart] = React.useState(0)
  const [regionEnd, setRegionEnd] = React.useState(0)
  const [currentTime, setCurrentTime] = React.useState(0)
  const [error, setError] = React.useState<string | null>(null)

  const selectedDuration = regionEnd - regionStart

  // Initialize WaveSurfer
  React.useEffect(() => {
    if (!containerRef.current || !file) return

    setIsReady(false)
    setError(null)

    const regions = RegionsPlugin.create()
    regionsRef.current = regions

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgb(148, 163, 184)",
      progressColor: "rgb(59, 130, 246)",
      cursorColor: "rgb(59, 130, 246)",
      height: 120,
      normalize: true,
      plugins: [regions],
    })

    wavesurferRef.current = ws

    ws.on("ready", () => {
      const audioDuration = ws.getDuration()
      setDuration(audioDuration)
      setIsReady(true)

      // Create initial region (max 30 seconds from start)
      const initialEnd = Math.min(audioDuration, MAX_DURATION)
      setRegionStart(0)
      setRegionEnd(initialEnd)

      regions.addRegion({
        start: 0,
        end: initialEnd,
        color: "rgba(59, 130, 246, 0.3)",
        drag: true,
        resize: true,
      })
    })

    ws.on("timeupdate", (time) => {
      setCurrentTime(time)
    })

    ws.on("play", () => setIsPlaying(true))
    ws.on("pause", () => setIsPlaying(false))
    ws.on("finish", () => setIsPlaying(false))

    ws.on("error", (err) => {
      console.error("WaveSurfer error:", err)
      setError("Failed to load audio file")
    })

    // Handle region updates
    regions.on("region-updated", (region) => {
      let start = region.start
      let end = region.end

      // Enforce max duration
      if (end - start > MAX_DURATION) {
        end = start + MAX_DURATION
        region.setOptions({ end })
      }

      setRegionStart(start)
      setRegionEnd(end)
    })

    // Load the file
    const url = URL.createObjectURL(file)
    ws.load(url)

    return () => {
      URL.revokeObjectURL(url)
      ws.destroy()
    }
  }, [file])

  const handlePlayPause = () => {
    if (!wavesurferRef.current) return

    if (isPlaying) {
      wavesurferRef.current.pause()
    } else {
      // Play from region start if outside region
      if (currentTime < regionStart || currentTime >= regionEnd) {
        wavesurferRef.current.setTime(regionStart)
      }
      wavesurferRef.current.play()
    }
  }

  const handleReset = () => {
    if (!wavesurferRef.current || !regionsRef.current) return

    const initialEnd = Math.min(duration, MAX_DURATION)
    setRegionStart(0)
    setRegionEnd(initialEnd)

    // Clear and recreate region
    regionsRef.current.clearRegions()
    regionsRef.current.addRegion({
      start: 0,
      end: initialEnd,
      color: "rgba(59, 130, 246, 0.3)",
      drag: true,
      resize: true,
    })
  }

  const handleExtractAndSave = async () => {
    if (!file) return

    try {
      // Read the file as ArrayBuffer
      const arrayBuffer = await file.arrayBuffer()
      const audioContext = new AudioContext()
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

      // Calculate sample positions
      const sampleRate = audioBuffer.sampleRate
      const startSample = Math.floor(regionStart * sampleRate)
      const endSample = Math.floor(regionEnd * sampleRate)
      const length = endSample - startSample

      // Create new buffer with trimmed audio
      const trimmedBuffer = audioContext.createBuffer(
        audioBuffer.numberOfChannels,
        length,
        sampleRate
      )

      // Copy the selected region
      for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        const sourceData = audioBuffer.getChannelData(channel)
        const targetData = trimmedBuffer.getChannelData(channel)
        for (let i = 0; i < length; i++) {
          targetData[i] = sourceData[startSample + i]
        }
      }

      // Convert to WAV blob
      const wavBlob = audioBufferToWav(trimmedBuffer)

      // Create a new File object
      const trimmedFile = new File(
        [wavBlob],
        file.name.replace(/\.[^/.]+$/, "") + "_trimmed.wav",
        { type: "audio/wav" }
      )

      onTrimmedAudio(trimmedFile)
      await audioContext.close()
    } catch (err) {
      console.error("Failed to extract audio:", err)
      setError("Failed to extract audio segment")
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 10)
    return `${mins}:${secs.toString().padStart(2, "0")}.${ms}`
  }

  if (error) {
    return (
      <div className={cn("p-4 rounded-lg border border-destructive/50 bg-destructive/10", className)}>
        <p className="text-sm text-destructive">{error}</p>
      </div>
    )
  }

  return (
    <div className={cn("space-y-3", className)}>
      {/* Waveform container */}
      <div className="rounded-lg border bg-muted/30 p-4">
        <div ref={containerRef} className="w-full" />

        {!isReady && (
          <div className="flex items-center justify-center h-[120px]">
            <p className="text-sm text-muted-foreground">Loading audio...</p>
          </div>
        )}
      </div>

      {/* Controls */}
      {isReady && (
        <>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Selection: {formatTime(regionStart)} - {formatTime(regionEnd)}</span>
            <span className={cn(
              "font-medium",
              selectedDuration > MAX_DURATION ? "text-destructive" : "text-primary"
            )}>
              Duration: {formatTime(selectedDuration)} / {MAX_DURATION}s max
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handlePlayPause}
            >
              {isPlaying ? (
                <Pause className="w-4 h-4 mr-1" />
              ) : (
                <Play className="w-4 h-4 mr-1" />
              )}
              {isPlaying ? "Pause" : "Preview"}
            </Button>

            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleReset}
            >
              <RotateCcw className="w-4 h-4 mr-1" />
              Reset
            </Button>

            <div className="flex-1" />

            <Button
              type="button"
              size="sm"
              onClick={handleExtractAndSave}
              disabled={selectedDuration > MAX_DURATION || selectedDuration < 1}
            >
              <Scissors className="w-4 h-4 mr-1" />
              Use Selection
            </Button>
          </div>

          {duration > MAX_DURATION && (
            <p className="text-xs text-amber-600">
              Audio is longer than {MAX_DURATION}s. Drag the highlighted region to select which part to use.
            </p>
          )}
        </>
      )}
    </div>
  )
}

// Helper function to convert AudioBuffer to WAV Blob
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels
  const sampleRate = buffer.sampleRate
  const format = 1 // PCM
  const bitDepth = 16

  const bytesPerSample = bitDepth / 8
  const blockAlign = numChannels * bytesPerSample

  const dataLength = buffer.length * blockAlign
  const bufferLength = 44 + dataLength

  const arrayBuffer = new ArrayBuffer(bufferLength)
  const view = new DataView(arrayBuffer)

  // WAV header
  writeString(view, 0, "RIFF")
  view.setUint32(4, 36 + dataLength, true)
  writeString(view, 8, "WAVE")
  writeString(view, 12, "fmt ")
  view.setUint32(16, 16, true) // fmt chunk size
  view.setUint16(20, format, true)
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * blockAlign, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitDepth, true)
  writeString(view, 36, "data")
  view.setUint32(40, dataLength, true)

  // Write interleaved audio data
  const offset = 44
  const channels: Float32Array[] = []
  for (let i = 0; i < numChannels; i++) {
    channels.push(buffer.getChannelData(i))
  }

  let pos = offset
  for (let i = 0; i < buffer.length; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channels[ch][i]))
      const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff
      view.setInt16(pos, int16, true)
      pos += 2
    }
  }

  return new Blob([arrayBuffer], { type: "audio/wav" })
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i))
  }
}
