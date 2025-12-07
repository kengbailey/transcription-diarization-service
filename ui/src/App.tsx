import { useState } from "react"
import { Mic, Users, Settings } from "lucide-react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { SpeakersTab } from "@/components/speakers/SpeakersTab"
import { TranscriptionTab } from "@/components/transcription/TranscriptionTab"
import { SettingsTab } from "@/components/layout/SettingsTab"

function App() {
  const [activeTab, setActiveTab] = useState("transcribe")

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center px-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
              <Mic className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-lg font-semibold">Speaker Diarization Studio</h1>
              <p className="text-xs text-muted-foreground">Transcribe & identify speakers</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="container px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-6">
            <TabsTrigger value="transcribe">
              <Mic className="w-4 h-4 mr-2" />
              Transcribe
            </TabsTrigger>
            <TabsTrigger value="speakers">
              <Users className="w-4 h-4 mr-2" />
              Speakers
            </TabsTrigger>
            <TabsTrigger value="settings">
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="transcribe">
            <TranscriptionTab />
          </TabsContent>

          <TabsContent value="speakers">
            <SpeakersTab />
          </TabsContent>

          <TabsContent value="settings">
            <SettingsTab />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-6 mt-auto">
        <div className="container px-4 text-center text-sm text-muted-foreground">
          Speaker Diarization Studio • Powered by pyannote.audio & Whisper
        </div>
      </footer>
    </div>
  )
}

export default App
