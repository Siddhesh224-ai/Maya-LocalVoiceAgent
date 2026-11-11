
import { useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import "./App.css";

const API_URL = "http://localhost:8000";
const preparationStages = [
  "Opening a private room",
  "Warming Maya’s voice",
  "Loading conversation memory",
];

function App() {
  const roomRef = useRef(null);
  const audioContainerRef = useRef(null);
  const responseStartedAtRef = useRef(null);

  const [status, setStatus] = useState("Idle");
  const [isMicOn, setIsMicOn] = useState(false);
  const [error, setError] = useState("");
  const [captions, setCaptions] = useState({ user: "", agent: "" });
  const [responseTime, setResponseTime] = useState(null);
  const [audioPlaybackReady, setAudioPlaybackReady] = useState(false);
  const [connectionDetail, setConnectionDetail] = useState("");
  const [preparationStage, setPreparationStage] = useState(0);

  const isPreparing = status === "Preparing";
  const isJoining = status === "Joining";
  const isBusy = isPreparing || isJoining;
  const isConnected = status === "Connected";

  useEffect(() => {
    if (!isPreparing) return undefined;

    const timer = window.setInterval(() => {
      setPreparationStage((stage) => (stage + 1) % preparationStages.length);
    }, 1600);

    return () => window.clearInterval(timer);
  }, [isPreparing]);

  // Start the voice call
  const startCall = async () => {
    try {
      setError("");
      setStatus("Preparing");
      setPreparationStage(0);
      setCaptions({ user: "", agent: "" });
      setResponseTime(null);
      setAudioPlaybackReady(false);
      setConnectionDetail("Preparing your private voice room…");
      responseStartedAtRef.current = null;

      // 1. Request a new call from FastAPI
      const response = await fetch(
        `${API_URL}/api/calls/start`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error("Backend could not start the call");
      }

      // 2. Receive LiveKit credentials
      const data = await response.json();

      const { livekit_url, token } = data;
      setStatus("Joining");
      setConnectionDetail("Maya is ready — joining your room…");

      // 3. Create LiveKit room client
      const microphoneOptions = {
        autoGainControl: true,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        voiceIsolation: true,
      };
      const room = new Room({
        audioCaptureDefaults: microphoneOptions,
      });

      roomRef.current = room;

      // 4. Listen for agent audio
      room.on(
        RoomEvent.TrackSubscribed,
        (track, publication, participant) => {
          if (track.kind === Track.Kind.Audio) {
            const element = track.attach();
            element.autoplay = true;
            element.setAttribute("playsinline", "true");
            audioContainerRef.current?.appendChild(element);
            element.play().catch(() => {
              setAudioPlaybackReady(false);
              setError("Audio is blocked by the browser. Click Enable Audio.");
            });
            setConnectionDetail(`Agent audio track subscribed (${participant.identity})`);
          }
        }
      );

      room.on(
        RoomEvent.TranscriptionReceived,
        (segments, participant) => {
          const latest = [...segments].reverse().find((segment) => segment.text?.trim());
          if (!latest?.text) return;

          const speaker = participant?.isLocal || participant?.identity === room.localParticipant.identity
            ? "user"
            : "agent";
          setCaptions((current) => ({ ...current, [speaker]: latest.text }));
          setConnectionDetail(`${speaker === "user" ? "You" : "Maya"} transcript received`);

          if (speaker === "user" && latest.final) {
            responseStartedAtRef.current = performance.now();
            setResponseTime(null);
          }

          if (speaker === "agent" && responseStartedAtRef.current !== null) {
            setResponseTime(
              (performance.now() - responseStartedAtRef.current) / 1000
            );
            responseStartedAtRef.current = null;
          }
        }
      );

      room.on(RoomEvent.ConnectionStateChanged, (state) => {
        setConnectionDetail(`LiveKit connection: ${state}`);
      });

      room.on(RoomEvent.ParticipantConnected, (participant) => {
        setConnectionDetail(`Agent joined (${participant.identity})`);
      });

      room.on(RoomEvent.AudioPlaybackStatusChanged, (playing) => {
        setAudioPlaybackReady(playing);
        if (!playing) {
          setError("Audio playback is blocked. Click Enable Audio to hear the agent.");
        } else {
          setError("");
        }
      });

      room.on(RoomEvent.MediaDevicesError, (deviceError) => {
        setIsMicOn(false);
        setError(`Microphone error: ${deviceError.message}`);
      });

      // 5. Remove audio when track is unsubscribed
      room.on(
        RoomEvent.TrackUnsubscribed,
        (track) => {
          track.detach().forEach((element) => element.remove());
        }
      );

      // 6. Handle disconnection
      room.on(RoomEvent.Disconnected, () => {
        setStatus("Idle");
        setIsMicOn(false);
        roomRef.current = null;
        responseStartedAtRef.current = null;
        setAudioPlaybackReady(false);
        setConnectionDetail("");
      });

      // 7. Connect browser to LiveKit
      await room.connect(livekit_url, token);

      // Explicitly unlock Web Audio for Safari and other autoplay-restricted browsers.
      try {
        await room.startAudio();
        setAudioPlaybackReady(true);
      } catch {
        setAudioPlaybackReady(false);
      }

      // 8. Enable microphone
      await room.localParticipant.setMicrophoneEnabled(true, microphoneOptions);

      setIsMicOn(true);
      setStatus("Connected");
      setConnectionDetail("Connected — Maya is listening");

    } catch (err) {
      console.error("Call error:", err);

      setError(err.message || "Something went wrong");
      setStatus("Idle");

      if (roomRef.current) {
        await roomRef.current.disconnect();
        roomRef.current = null;
      }
    }
  };

  // End the call
  const endCall = async () => {
    if (roomRef.current) {
      await roomRef.current.disconnect();
      roomRef.current = null;
    }

    setIsMicOn(false);
    setStatus("Idle");
    setAudioPlaybackReady(false);
    setConnectionDetail("");
  };

  const enableAudio = async () => {
    if (!roomRef.current) return;

    try {
      await roomRef.current.startAudio();
      setAudioPlaybackReady(true);
      setError("");
    } catch (err) {
      setError(`Could not enable audio: ${err.message}`);
    }
  };

  // Toggle microphone
  const toggleMic = async () => {
    const room = roomRef.current;

    if (!room) return;

    try {
      const newMicState = !isMicOn;

      await room.localParticipant.setMicrophoneEnabled(
        newMicState
      );

      setIsMicOn(newMicState);

    } catch (err) {
      console.error("Microphone error:", err);
      setError("Could not change microphone state");
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">m</span>
          <span>Maya voice</span>
        </div>
        <span className="secure-label"><span className="secure-dot" /> Private session</span>
      </header>

      <main className="voice-card">
        <section className="hero-panel">
          <div className="eyebrow">LOCAL INTELLIGENCE <span>•</span> HUMAN CONVERSATION</div>
          <div className={`voice-orb ${isPreparing ? "is-loading" : ""} ${isConnected ? "is-live" : ""}`} aria-hidden="true">
            <div className="orb-core">M</div>
            <span className="orb-ring orb-ring-one" />
            <span className="orb-ring orb-ring-two" />
          </div>
          <p className="agent-name">Maya</p>
          <h1>Your thoughtful voice companion.</h1>
          <p className="hero-copy">
            Ask a question, think out loud, or pick up where you left off.
          </p>

          <div className={`status-pill ${isConnected ? "live" : ""} ${isBusy ? "busy" : ""}`} role="status" aria-live="polite">
            <span className="status-dot" />
            <span>{isConnected ? "Listening now" : isBusy ? "Getting ready" : "Ready when you are"}</span>
          </div>

          {isPreparing && (
            <div className="preparation-panel" role="status" aria-live="polite" aria-busy="true">
              <div className="preparation-heading">
                <span className="loader" aria-hidden="true" />
                <div>
                  <strong>Preparing your conversation</strong>
                  <span>{preparationStages[preparationStage]}…</span>
                </div>
              </div>
              <div className="preparation-progress" aria-hidden="true">
                {preparationStages.map((stage, index) => (
                  <span key={stage} className={index <= preparationStage ? "complete" : ""} />
                ))}
              </div>
              <p>This usually takes a moment. You’ll join as soon as Maya is fully warmed up.</p>
            </div>
          )}

          {connectionDetail && <p className="connection-detail">{connectionDetail}</p>}
        </section>

        <section className="conversation-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">CONVERSATION</span>
              <h2>Live captions</h2>
            </div>
            <span className="latency-badge">Low latency</span>
          </div>

          <div className="captions" aria-label="Live captions" aria-live="polite">
            <div className="caption caption-user">
              <span className="caption-label"><i /> You</span>
              <p>{captions.user || "Your words will appear here"}</p>
            </div>
            <div className="caption caption-agent">
              <span className="caption-label"><i /> Maya</span>
              <p>{captions.agent || "Maya’s response will appear here"}</p>
            </div>
          </div>

          {responseTime !== null && (
            <p className="response-time">Response started in {responseTime.toFixed(1)}s</p>
          )}

          <div className="controls">
            <button
              className="start-btn"
              onClick={startCall}
              disabled={isBusy || isConnected}
              aria-busy={isBusy}
            >
              <span>{isBusy ? "Preparing Maya" : isConnected ? "Call in progress" : "Start a conversation"}</span>
              {!isBusy && !isConnected && <span className="button-arrow" aria-hidden="true">↗</span>}
              {isBusy && <span className="button-spinner" aria-hidden="true" />}
            </button>

            <div className="secondary-controls">
              <button className={`utility-btn ${isMicOn ? "selected" : ""}`} onClick={toggleMic} disabled={!isConnected}>
                <span className="button-icon" aria-hidden="true">⌁</span>{isMicOn ? "Mic on" : "Mic off"}
              </button>
              <button className={`utility-btn ${audioPlaybackReady ? "selected" : ""}`} onClick={enableAudio} disabled={!isConnected || audioPlaybackReady}>
                <span className="button-icon" aria-hidden="true">◖</span>{audioPlaybackReady ? "Audio on" : "Enable audio"}
              </button>
              <button className="utility-btn end-btn" onClick={endCall} disabled={!isConnected}>
                End call
              </button>
            </div>
          </div>

          {error && <p className="error" role="alert">{error}</p>}
          <div ref={audioContainerRef} id="audio-container" />
        </section>
      </main>

      <footer className="footer-note">
        <span>Powered by your private local voice stack</span>
        <span className="footer-line" />
        <span>Ready in seconds</span>
      </footer>
    </div>
  );
}

export default App;
