import * as React from "react"
import { Server, Database, Cpu, CheckCircle, XCircle, RefreshCw, Users, AudioLines } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { getHealth, getStats, type HealthResponse, type StatsResponse } from "@/lib/api"
import { cn } from "@/lib/utils"

export function SettingsTab() {
  const [health, setHealth] = React.useState<HealthResponse | null>(null)
  const [stats, setStats] = React.useState<StatsResponse | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const loadData = React.useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const [healthData, statsData] = await Promise.all([
        getHealth(),
        getStats(),
      ])
      setHealth(healthData)
      setStats(statsData)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load system status")
    } finally {
      setLoading(false)
    }
  }, [])

  React.useEffect(() => {
    loadData()
  }, [loadData])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Settings & Status</h2>
          <p className="text-muted-foreground">
            System status and configuration
          </p>
        </div>
        <Button variant="outline" onClick={loadData} disabled={loading}>
          <RefreshCw className={cn("w-4 h-4 mr-2", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* Error message */}
      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
          {error}
        </div>
      )}

      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="w-5 h-5" />
            System Status
          </CardTitle>
          <CardDescription>
            Current status of the Speaker Diarization API
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <StatusItem
              label="API Status"
              value={health?.status || "Unknown"}
              status={health?.status === "healthy" ? "success" : "error"}
            />
            <StatusItem
              label="Models Loaded"
              value={health?.models_loaded ? "Yes" : "No"}
              status={health?.models_loaded ? "success" : "warning"}
            />
            <StatusItem
              label="Qdrant Connected"
              value={health?.qdrant_connected ? "Yes" : "No"}
              status={health?.qdrant_connected ? "success" : "error"}
            />
            <StatusItem
              label="Compute Device"
              value={health?.device?.toUpperCase() || "Unknown"}
              status="neutral"
              icon={<Cpu className="w-4 h-4" />}
            />
          </div>
        </CardContent>
      </Card>

      {/* Database Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            Database Statistics
          </CardTitle>
          <CardDescription>
            Speaker database information
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="flex items-center gap-4 p-4 rounded-lg bg-muted">
              <div className="p-3 rounded-full bg-primary/10">
                <Users className="w-6 h-6 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Speakers</p>
                <p className="text-2xl font-bold">{stats?.speakers?.total_count || 0}</p>
              </div>
            </div>
            <div className="flex items-center gap-4 p-4 rounded-lg bg-muted">
              <div className="p-3 rounded-full bg-emerald-500/10">
                <AudioLines className="w-6 h-6 text-emerald-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Embeddings</p>
                <p className="text-2xl font-bold">{stats?.speakers?.total_embeddings || 0}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* API Info */}
      <Card>
        <CardHeader>
          <CardTitle>API Information</CardTitle>
          <CardDescription>
            API version and connection details
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Version</p>
              <p className="font-medium">{health?.version || "Unknown"}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">API Endpoint</p>
              <p className="font-medium font-mono text-sm">/api (proxied to localhost:8000)</p>
            </div>
          </div>
          
          <div className="pt-4 border-t">
            <h4 className="font-medium mb-2">Quick Links</h4>
            <div className="flex flex-wrap gap-2">
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium h-8 px-3 border border-input bg-transparent shadow-sm hover:bg-accent hover:text-accent-foreground transition-colors"
              >
                API Documentation
              </a>
              <a
                href="http://localhost:8000/health"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium h-8 px-3 border border-input bg-transparent shadow-sm hover:bg-accent hover:text-accent-foreground transition-colors"
              >
                Health Check
              </a>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* About */}
      <Card>
        <CardHeader>
          <CardTitle>About</CardTitle>
        </CardHeader>
        <CardContent className="prose prose-invert max-w-none">
          <p className="text-muted-foreground">
            Speaker Diarization Studio is a modern UI for the Speaker Diarization API. 
            It allows you to register speakers, manage voice samples, and transcribe 
            audio with speaker identification.
          </p>
          <div className="mt-4 text-sm text-muted-foreground">
            <p>Built with:</p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>pyannote.audio - Speaker diarization</li>
              <li>wespeaker - Speaker embeddings</li>
              <li>Qdrant - Vector database</li>
              <li>Whisper API - Transcription</li>
              <li>React + TailwindCSS - Frontend</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

interface StatusItemProps {
  label: string
  value: string
  status: "success" | "error" | "warning" | "neutral"
  icon?: React.ReactNode
}

function StatusItem({ label, value, status, icon }: StatusItemProps) {
  const statusColors = {
    success: "text-emerald-500",
    error: "text-red-500",
    warning: "text-yellow-500",
    neutral: "text-muted-foreground",
  }

  const StatusIcon = () => {
    if (icon) return <>{icon}</>
    switch (status) {
      case "success":
        return <CheckCircle className="w-4 h-4" />
      case "error":
        return <XCircle className="w-4 h-4" />
      default:
        return null
    }
  }

  return (
    <div className="p-4 rounded-lg bg-muted">
      <p className="text-sm text-muted-foreground mb-1">{label}</p>
      <div className={cn("flex items-center gap-2 font-medium", statusColors[status])}>
        <StatusIcon />
        <span>{value}</span>
      </div>
    </div>
  )
}
