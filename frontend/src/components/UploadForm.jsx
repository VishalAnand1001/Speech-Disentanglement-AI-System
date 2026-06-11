import { useState, useRef } from "react";

function UploadForm() {
  const [mixedAudio, setMixedAudio]     = useState(null);
  const [scammerAudio, setScammerAudio] = useState(null);
  const [keyword, setKeyword]           = useState("");
  const [loading, setLoading]           = useState(false);
  const [result, setResult]             = useState(null);
  const audioRef = useRef(null);

  const jumpToTimestamp = (ts) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = ts;
    audioRef.current.play();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!mixedAudio || !scammerAudio || !keyword) { alert("Please fill all fields."); return; }
    try {
      setLoading(true); setResult(null);
      const fd = new FormData();
      fd.append("mixedAudio", mixedAudio);
      fd.append("scammerAudio", scammerAudio);
      fd.append("keyword", keyword);
      const res  = await fetch("http://127.0.0.1:5000/analyze", { method: "POST", body: fd });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err); alert("Error running analysis.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="shell">

      {/* ── LEFT: inputs ── */}
      <div className="panel panel--left">
        <div className="topbar">
          <div>
            <h1 className="title">Speech<span className="highlight">Disentangle</span></h1>
            <p className="subtitle">Scammer Voice Isolation & Keyword Detection</p>
          </div>
          <div className="status"><span className="status__dot" />SECURE</div>
        </div>

        <div className="divider" />

        <form onSubmit={handleSubmit} style={{ display:"flex", flexDirection:"column", gap:"14px", flex:1 }}>

          <div className="section">
            <label>Mixed Audio Recording</label>
            <input type="file" accept=".wav,.mp3"
              onChange={(e) => setMixedAudio(e.target.files[0])} />
          </div>

          <div className="section">
            <label>Scammer Voice Sample</label>
            <input type="file" accept=".wav,.mp3"
              onChange={(e) => setScammerAudio(e.target.files[0])} />
          </div>

          <div className="section">
            <label>Keyword To Detect</label>
            <input type="text"
              placeholder="e.g. OTP, Password, Bank Account"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)} />
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "⟳  ANALYZING..." : "▶  ANALYZE RECORDING"}
          </button>

        </form>
      </div>

      {/* ── RIGHT: results ── */}
      <div className="panel panel--right">
        <div className="results-header">
          <span className="results-title">RESULTS</span>
          {result && (
            <span className={`found-badge ${result.found ? "found-badge--yes" : "found-badge--no"}`}>
              {result.found ? "● KEYWORD FOUND" : "● NOT FOUND"}
            </span>
          )}
        </div>

        <div className="divider" />

        {!result && !loading && (
          <div className="empty-state">
            <div className="empty-icon">⬡</div>
            <p>Run an analysis to see results here</p>
          </div>
        )}

        {loading && (
          <div className="empty-state">
            <div className="spinner" />
            <p>Processing audio…</p>
          </div>
        )}

        {result && (
          <div className="results-body">
            <audio ref={audioRef} controls className="audio-player">
              <source src="http://127.0.0.1:5000/audio" type="audio/wav" />
            </audio>

            {result.matches && result.matches.length > 0 ? (
              <div className="matches-list">
                {result.matches.map((match, i) => (
                  <div key={i} className="match-card">
                    <div className="match-card__top">
                      <span className="match-word">{match.detected_as}</span>
                      <button type="button" className="jump-btn"
                        onClick={() => jumpToTimestamp(match.start)}>
                        ↳ Jump
                      </button>
                    </div>
                    <div className="match-card__meta">
                      <span className="meta-item">
                        <span className="meta-label">TIME</span>{match.start.toFixed(2)}s
                      </span>
                      <span className="meta-item">
                        <span className="meta-label">CONF</span>{(match.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    {match.segment_text && (
                      <p className="match-context">"{match.segment_text}"</p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-matches">No keyword occurrences detected.</p>
            )}
          </div>
        )}
      </div>

    </div>
  );
}

export default UploadForm;