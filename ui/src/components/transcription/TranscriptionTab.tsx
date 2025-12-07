import * as React from "react"
import { Play, Pause, Upload, FileAudio, Clock, Users, Languages, Copy, Check } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input, Label } from "@/components/ui/input"
import { FileUpload } from "@/components/ui/file-upload"
import { Spinner, LoadingOverlay } from "@/components/ui/spinner"
import { transcribeIdentified, type TranscriptionResult, type TranscriptSegment } from "@/lib/api"
import { cn, formatTime, formatDuration, getSpeakerColor } from "@/lib/utils"

export function TranscriptionTab() {
  const [audioFile, setAudioFile] = React.useState<File | null>(null)
  const [audioUrl, setAudioUrl] = React.useState<string | null>(null)
  const [numSpeakers, setNumSpeakers] = React.useState<string>("")
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [result, setResult] = React.useState<TranscriptionResult | null>(null)
  
  // Audio playback
  const audioRef = React.useRef<HTMLAudioElement>(null)
  const [isPlaying, setIsPlaying] = React.useState(false)
  const [currentTime, setCurrentTime] = React.useState(0)
  const [duration, setDuration] = React.useState(0)
  
  // Copy state
  const [copied, setCopied] = React.useState(false)

  // Build speaker index mapping for colors
  const speakerIndexMap = React.useMemo(() => {
    if (!result) return new Map<string, number>()
    const uniqueSpeakers = [...new Set(result.segments.map(s => s.speaker))]
    return new Map(uniqueSpeakers.map((speaker, index) => [speaker, index]))
  }, [result])

  React.useEffect(() => {
    if (audioFile) {
      const url = URL.createObjectURL(audioFile)
      setAudioUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setAudioUrl(null)
    }
  }, [audioFile])

  const handleTranscribe = async () => {
    if (!audioFile) return
    
    try {
      setLoading(true)
      setError(null)
      const speakers = numSpeakers ? parseInt(numSpeakers, 10) : undefined
      const transcription = await transcribeIdentified(audioFile, speakers)
      setResult(transcription)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcription failed")
    } finally {
      setLoading(false)
    }
  }

  const handlePlayPause = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  const handleSeek = (time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time
      setCurrentTime(time)
    }
  }

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const percent = (e.clientX - rect.left) / rect.width
    handleSeek(percent * duration)
  }

  const handleCopyTranscript = async () => {
    if (!result) return
    
    const text = result.segments
      .map(s => `[${s.identified_as || s.speaker}] ${s.text}`)
      .join("\n\n")
    
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleReset = () => {
    setAudioFile(null)
    setResult(null)
    setError(null)
    setCurrentTime(0)
    setDuration(0)
    setIsPlaying(false)
  }

  // Get the active segment based on current time
  const activeSegmentIndex = React.useMemo(() => {
    if (!result) return -1
    return result.segments.findIndex(
      s => currentTime >= s.start && currentTime <= s.end
    )
  }, [result, currentTime])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Audio Transcription</h2>
        <p className="text-muted-foreground">
          Upload audio to transcribe with speaker identification
        </p>
      </div>

      {/* Error message */}
      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
          {error}
        </div>
      )}

      {!result ? (
        /* Upload & Configure */
        <Card className="relative">
          {loading && <LoadingOverlay message="Transcribing audio... This may take a few minutes." />}
          <CardContent className="p-6 space-y-6">
            <div className="space-y-2">
              <Label>Audio File</Label>
              <FileUpload
                value={audioFile}
                onChange={setAudioFile}
                disabled={loading}
              />
            </div>

            {audioUrl && (
              <div className="space-y-2">
                <Label>Preview</Label>
                <audio
                  ref={audioRef}
                  src={audioUrl}
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onEnded={() => setIsPlaying(false)}
                  className="hidden"
                />
                <div className="flex items-center gap-4 p-4 bg-muted rounded-lg">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handlePlayPause}
                    className="shrink-0"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </Button>
                  <div
                    className="flex-1 h-2 bg-background rounded-full cursor-pointer"
                    onClick={handleProgressClick}
                  >
                    <div
                      className="h-full bg-primary rounded-full transition-all"
                      style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
                    />
                  </div>
                  <span className="text-sm text-muted-foreground shrink-0 font-mono">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </span>
                </div>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="num-speakers">Number of Speakers (optional)</Label>
              <Input
                id="num-speakers"
                type="number"
                min={1}
                max={10}
                placeholder="Auto-detect"
                value={numSpeakers}
                onChange={(e) => setNumSpeakers(e.target.value)}
                disabled={loading}
                className="max-w-[200px]"
              />
              <p className="text-xs text-muted-foreground">
                Leave empty to auto-detect the number of speakers.
              </p>
            </div>

            <Button
              size="lg"
              onClick={handleTranscribe}
              disabled={!audioFile || loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Spinner size="sm" className="mr-2" />
                  Transcribing...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Transcribe Audio
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      ) : (
        /* Results */
        <div className="space-y-6">
          {/* Audio Player */}
          <Card>
            <CardContent className="p-4">
              <audio
                ref={audioRef}
                src={audioUrl || undefined}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={() => setIsPlaying(false)}
                className="hidden"
              />
              <div className="flex items-center gap-4">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handlePlayPause}
                  className="shrink-0"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </Button>
                
                {/* Timeline with segments */}
                <div
                  className="flex-1 h-8 bg-muted rounded-lg cursor-pointer relative overflow-hidden"
                  onClick={handleProgressClick}
                >
                  {/* Segment colors */}
                  {result.segments.map((segment, idx) => {
                    const speakerIdx = speakerIndexMap.get(segment.speaker) || 0
                    const color = getSpeakerColor(speakerIdx)
                    const left = (segment.start / duration) * 100
                    const width = ((segment.end - segment.start) / duration) * 100
                    return (
                      <div
                        key={idx}
                        className={cn("absolute top-0 h-full", color.bg)}
                        style={{
                          left: `${left}%`,
                          width: `${width}%`,
                          backgroundColor: color.solid,
                          opacity: 0.3,
                        }}
                      />
                    )
                  })}
                  {/* Progress indicator */}
                  <div
                    className="absolute top-0 h-full w-0.5 bg-white shadow-lg z-10"
                    style={{ left: `${duration ? (currentTime / duration) * 100 : 0}%` }}
                  />
                </div>
                
                <span className="text-sm text-muted-foreground shrink-0 font-mono">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Stats */}
          <div className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardContent className="p-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Clock className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Duration</p>
                  <p className="font-semibold">{formatDuration(result.duration)}</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-500/10">
                  <Users className="w-5 h-5 text-emerald-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Speakers</p>
                  <p className="font-semibold">{result.num_speakers} ({result.num_identified} identified)</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-500/10">
                  <Languages className="w-5 h-5 text-purple-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Language</p>
                  <p className="font-semibold">{result.language?.toUpperCase() || "Unknown"}</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-orange-500/10">
                  <FileAudio className="w-5 h-5 text-orange-500" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Segments</p>
                  <p className="font-semibold">{result.segments.length}</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Transcript */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle>Transcript</CardTitle>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={handleCopyTranscript}>
                  {copied ? (
                    <>
                      <Check className="w-4 h-4 mr-1" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4 mr-1" />
                      Copy
                    </>
                  )}
                </Button>
                <Button variant="outline" size="sm" onClick={handleReset}>
                  New Transcription
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3 max-h-[500px] overflow-y-auto">
              {result.segments.map((segment, idx) => (
                <TranscriptSegmentItem
                  key={idx}
                  segment={segment}
                  speakerIndex={speakerIndexMap.get(segment.speaker) || 0}
                  isActive={idx === activeSegmentIndex}
                  onClick={() => handleSeek(segment.start)}
                />
              ))}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

interface TranscriptSegmentItemProps {
  segment: TranscriptSegment
  speakerIndex: number
  isActive: boolean
  onClick: () => void
}

function TranscriptSegmentItem({ segment, speakerIndex, isActive, onClick }: TranscriptSegmentItemProps) {
  const color = getSpeakerColor(speakerIndex)
  const displayName = segment.identified_as || segment.speaker

  return (
    <div
      onClick={onClick}
      className={cn(
        "p-4 rounded-lg cursor-pointer transition-all border",
        isActive
          ? `${color.bg} ${color.border} border`
          : "bg-muted/30 border-transparent hover:bg-muted/50"
      )}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className={cn(
          "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium",
          color.bg, color.text
        )}>
          {displayName}
        </span>
        <span className="text-xs text-muted-foreground font-mono">
          {formatTime(segment.start)} - {formatTime(segment.end)}
        </span>
        {segment.confidence !== null && segment.confidence > 0 && (
          <span className="text-xs text-muted-foreground">
            ({Math.round(segment.confidence * 100)}% confident)
          </span>
        )}
      </div>
      <p className={cn("text-sm", isActive ? color.text : "text-foreground")}>
        {segment.text}
      </p>
    </div>
  )
}
