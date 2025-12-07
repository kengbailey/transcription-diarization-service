import * as React from "react"
import { Plus, Users, Trash2, AudioLines, Calendar } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog"
import { Input, Label } from "@/components/ui/input"
import { FileUpload } from "@/components/ui/file-upload"
import { Spinner } from "@/components/ui/spinner"
import { getSpeakers, registerSpeaker, addSpeakerSample, deleteSpeaker, type Speaker } from "@/lib/api"
import { getSpeakerColor } from "@/lib/utils"

export function SpeakersTab() {
  const [speakers, setSpeakers] = React.useState<Speaker[]>([])
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  
  // Add speaker dialog
  const [addDialogOpen, setAddDialogOpen] = React.useState(false)
  const [newSpeakerName, setNewSpeakerName] = React.useState("")
  const [newSpeakerFile, setNewSpeakerFile] = React.useState<File | null>(null)
  const [addingLoading, setAddingLoading] = React.useState(false)
  
  // Add sample dialog
  const [sampleDialogOpen, setSampleDialogOpen] = React.useState(false)
  const [selectedSpeaker, setSelectedSpeaker] = React.useState<Speaker | null>(null)
  const [sampleFile, setSampleFile] = React.useState<File | null>(null)
  const [sampleLoading, setSampleLoading] = React.useState(false)
  
  // Delete confirmation
  const [deleteDialogOpen, setDeleteDialogOpen] = React.useState(false)
  const [speakerToDelete, setSpeakerToDelete] = React.useState<Speaker | null>(null)
  const [deleteLoading, setDeleteLoading] = React.useState(false)

  const loadSpeakers = React.useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await getSpeakers()
      setSpeakers(response.speakers)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load speakers")
    } finally {
      setLoading(false)
    }
  }, [])

  React.useEffect(() => {
    loadSpeakers()
  }, [loadSpeakers])

  const handleAddSpeaker = async () => {
    if (!newSpeakerName.trim() || !newSpeakerFile) return
    
    try {
      setAddingLoading(true)
      await registerSpeaker(newSpeakerName.trim(), newSpeakerFile)
      setAddDialogOpen(false)
      setNewSpeakerName("")
      setNewSpeakerFile(null)
      loadSpeakers()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add speaker")
    } finally {
      setAddingLoading(false)
    }
  }

  const handleAddSample = async () => {
    if (!selectedSpeaker || !sampleFile) return
    
    try {
      setSampleLoading(true)
      await addSpeakerSample(selectedSpeaker.speaker_id, sampleFile)
      setSampleDialogOpen(false)
      setSelectedSpeaker(null)
      setSampleFile(null)
      loadSpeakers()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add sample")
    } finally {
      setSampleLoading(false)
    }
  }

  const handleDeleteSpeaker = async () => {
    if (!speakerToDelete) return
    
    try {
      setDeleteLoading(true)
      await deleteSpeaker(speakerToDelete.speaker_id)
      setDeleteDialogOpen(false)
      setSpeakerToDelete(null)
      loadSpeakers()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete speaker")
    } finally {
      setDeleteLoading(false)
    }
  }

  const openAddSampleDialog = (speaker: Speaker) => {
    setSelectedSpeaker(speaker)
    setSampleDialogOpen(true)
  }

  const openDeleteDialog = (speaker: Speaker) => {
    setSpeakerToDelete(speaker)
    setDeleteDialogOpen(true)
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
  }

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
          <h2 className="text-2xl font-bold">Registered Speakers</h2>
          <p className="text-muted-foreground">
            {speakers.length} speaker{speakers.length !== 1 ? "s" : ""} registered
          </p>
        </div>
        <Button onClick={() => setAddDialogOpen(true)}>
          <Plus className="w-4 h-4 mr-2" />
          Add Speaker
        </Button>
      </div>

      {/* Error message */}
      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
          {error}
        </div>
      )}

      {/* Speaker grid */}
      {speakers.length === 0 ? (
        <Card className="p-12">
          <div className="flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
              <Users className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">No speakers registered</h3>
            <p className="text-muted-foreground mb-4 max-w-sm">
              Register speakers to enable voice identification in your transcriptions.
            </p>
            <Button onClick={() => setAddDialogOpen(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add Your First Speaker
            </Button>
          </div>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {speakers.map((speaker, index) => {
            const color = getSpeakerColor(index)
            return (
              <Card key={speaker.speaker_id} className="overflow-hidden">
                <div className={`h-2 ${color.bg.replace('/20', '')}`} />
                <CardContent className="p-5">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${color.bg} ${color.text}`}>
                        <span className="font-semibold text-sm">
                          {speaker.speaker_name.slice(0, 2).toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <h3 className="font-semibold">{speaker.speaker_name}</h3>
                        <p className="text-xs text-muted-foreground">
                          ID: {speaker.speaker_id.slice(0, 8)}...
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-muted-foreground hover:text-destructive"
                      onClick={() => openDeleteDialog(speaker)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                  
                  <div className="space-y-2 text-sm text-muted-foreground mb-4">
                    <div className="flex items-center gap-2">
                      <AudioLines className="w-4 h-4" />
                      <span>{speaker.embeddings_count} voice sample{speaker.embeddings_count !== 1 ? "s" : ""}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4" />
                      <span>Added {formatDate(speaker.created_at)}</span>
                    </div>
                  </div>
                  
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() => openAddSampleDialog(speaker)}
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Sample
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}

      {/* Add Speaker Dialog */}
      <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
        <DialogContent onClose={() => setAddDialogOpen(false)}>
          <DialogHeader>
            <DialogTitle>Add New Speaker</DialogTitle>
            <DialogDescription>
              Register a new speaker by providing their name and a voice sample.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="speaker-name">Speaker Name</Label>
              <Input
                id="speaker-name"
                placeholder="e.g., John Doe"
                value={newSpeakerName}
                onChange={(e) => setNewSpeakerName(e.target.value)}
                disabled={addingLoading}
              />
            </div>
            
            <div className="space-y-2">
              <Label>Voice Sample</Label>
              <FileUpload
                value={newSpeakerFile}
                onChange={setNewSpeakerFile}
                disabled={addingLoading}
              />
              <p className="text-xs text-muted-foreground">
                Upload a clear audio recording of the speaker (10-30 seconds works best).
              </p>
            </div>
          </div>
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setAddDialogOpen(false)}
              disabled={addingLoading}
            >
              Cancel
            </Button>
            <Button
              onClick={handleAddSpeaker}
              disabled={!newSpeakerName.trim() || !newSpeakerFile || addingLoading}
            >
              {addingLoading && <Spinner size="sm" className="mr-2" />}
              Add Speaker
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add Sample Dialog */}
      <Dialog open={sampleDialogOpen} onOpenChange={setSampleDialogOpen}>
        <DialogContent onClose={() => setSampleDialogOpen(false)}>
          <DialogHeader>
            <DialogTitle>Add Voice Sample</DialogTitle>
            <DialogDescription>
              Add another voice sample for {selectedSpeaker?.speaker_name} to improve recognition accuracy.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <FileUpload
              value={sampleFile}
              onChange={setSampleFile}
              disabled={sampleLoading}
            />
            <p className="text-xs text-muted-foreground">
              Adding more samples helps improve speaker identification accuracy.
            </p>
          </div>
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setSampleDialogOpen(false)}
              disabled={sampleLoading}
            >
              Cancel
            </Button>
            <Button
              onClick={handleAddSample}
              disabled={!sampleFile || sampleLoading}
            >
              {sampleLoading && <Spinner size="sm" className="mr-2" />}
              Add Sample
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent onClose={() => setDeleteDialogOpen(false)}>
          <DialogHeader>
            <DialogTitle>Delete Speaker</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete {speakerToDelete?.speaker_name}? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
              disabled={deleteLoading}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteSpeaker}
              disabled={deleteLoading}
            >
              {deleteLoading && <Spinner size="sm" className="mr-2" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
