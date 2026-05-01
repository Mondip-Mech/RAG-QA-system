"""Custom CSS injected into Streamlit to get a modern AI-product look."""

CUSTOM_CSS = """
<style>
/* ---------- Global ---------- */
.stApp {
  background: #0b0d12;
  color: #e6e8ee;
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
}
section[data-testid="stSidebar"] {
  background: #11141b;
  border-right: 1px solid #1f232d;
}
section[data-testid="stSidebar"] * { color: #d8dbe3 !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 7rem; max-width: 980px; }

/* ---------- Hero ---------- */
.hero {
  background: linear-gradient(135deg, #1a1f2e 0%, #0f1320 100%);
  border: 1px solid #232838;
  border-radius: 16px;
  padding: 28px 32px;
  margin-bottom: 22px;
}
.hero h1 {
  font-size: 28px; font-weight: 600; margin: 0 0 8px 0;
  background: linear-gradient(90deg, #fff 0%, #9aa3b8 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #98a2b8; margin: 0 0 14px 0; font-size: 14px; }
.badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: #1a1f2e; border: 1px solid #2a3142;
  color: #b8c0d0; padding: 5px 12px; border-radius: 999px;
  font-size: 12px; font-weight: 500;
}
.badge .dot { width: 6px; height: 6px; border-radius: 50%; background: #4ade80; }

/* ---------- Sidebar widgets ---------- */
.sidebar-section {
  background: #161a23; border: 1px solid #1f232d;
  border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;
}
.sidebar-section .label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
  color: #6b7488; font-weight: 600; margin-bottom: 8px;
}
.runtime-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 0; font-size: 12.5px;
  gap: 10px;
}
.runtime-row .k { color: #8893aa; flex-shrink: 0; }
.runtime-row .v {
  color: #e6e8ee; font-weight: 500;
  text-align: right;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  min-width: 0;
}
.runtime-row .v.ok { color: #4ade80; }

.file-pill {
  display: flex; align-items: center; justify-content: space-between;
  background: #0f1320; border: 1px solid #232838;
  border-radius: 8px; padding: 8px 10px; margin-bottom: 6px;
  font-size: 12.5px;
}
.file-pill .name { color: #cfd6e4; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 160px; }
.file-pill .meta { color: #6b7488; font-size: 11px; }

/* ---------- Chat ---------- */
.chat-msg {
  display: flex; gap: 14px; padding: 16px 0;
  border-bottom: 1px solid #1a1f2e;
}
.chat-msg.user { background: transparent; }
.avatar {
  width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 600;
}
.avatar.user { background: #2a3142; color: #cfd6e4; }
.avatar.bot  { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; }
.msg-body { flex: 1; min-width: 0; }
.msg-role { font-size: 11px; color: #6b7488; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.msg-content { color: #e6e8ee; line-height: 1.65; font-size: 15px; }
.msg-content h3 {
  font-size: 12px; font-weight: 600; color: #8b9bb5;
  margin: 22px 0 10px 0; text-transform: uppercase; letter-spacing: 0.08em;
  border-bottom: 1px solid #1f232d; padding-bottom: 6px;
}
.msg-content h3:first-child { margin-top: 4px; }
.msg-content ul { margin: 6px 0 14px 22px; }
.msg-content li { margin-bottom: 6px; }
.msg-content p { margin: 0 0 12px 0; }
.msg-content ul { margin: 4px 0 8px 18px; }
.msg-content code { background: #1a1f2e; padding: 1px 6px; border-radius: 4px; font-size: 13px; }

/* Citation marker pill inside answer text */
.cite-marker {
  display: inline-flex; align-items: center; justify-content: center;
  background: #2a3142; color: #a5b4fc; border: 1px solid #3a4256;
  border-radius: 6px; padding: 1px 6px; margin: 0 2px;
  font-size: 11px; font-weight: 600;
}

/* Citation cards */
.cite-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px; margin-top: 14px;
}
.cite-card {
  background: #11141b; border: 1px solid #232838;
  border-radius: 10px; padding: 12px; transition: border-color 0.15s;
}
.cite-card:hover { border-color: #3a4256; }
.cite-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.cite-tag {
  background: #2a3142; color: #a5b4fc; border-radius: 4px;
  padding: 2px 7px; font-size: 11px; font-weight: 700;
}
.cite-source { font-size: 12px; color: #cfd6e4; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cite-page { font-size: 11px; color: #6b7488; }
.cite-snippet { font-size: 12px; color: #98a2b8; line-height: 1.5; max-height: 64px; overflow: hidden; }

/* Pipeline step chips (during streaming) */
.steps { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.step-chip {
  background: #11141b; border: 1px solid #232838;
  border-radius: 999px; padding: 4px 10px; font-size: 11.5px; color: #98a2b8;
}
.step-chip.done { color: #4ade80; border-color: #1f3a2e; }
.step-chip.active { color: #a5b4fc; border-color: #2e3856; }

/* Verification badge */
.verify {
  display: inline-flex; gap: 8px; align-items: center;
  background: #11141b; border: 1px solid #232838;
  border-radius: 8px; padding: 6px 10px; font-size: 11.5px; margin-top: 10px;
}
.verify .ok  { color: #4ade80; }
.verify .mid { color: #fbbf24; }
.verify .bad { color: #f87171; }

/* Mermaid diagram container */
iframe[title*="mermaid"] {
  background: #11141b !important;
  border: 1px solid #232838 !important;
  border-radius: 10px !important;
  margin: 12px 0 !important;
}
svg[id^="mermaid-"] {
  background: #11141b;
  border-radius: 10px;
  padding: 16px;
}

/* Hide some default streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Buttons */
.stButton > button {
  background: #1a1f2e; color: #e6e8ee; border: 1px solid #2a3142;
  border-radius: 8px; font-weight: 500; transition: all 0.15s;
}
.stButton > button:hover { background: #232838; border-color: #3a4256; }

/* Chat input float feel */
[data-testid="stChatInput"] {
  background: #11141b; border: 1px solid #232838; border-radius: 12px;
}
[data-testid="stChatInput"] textarea { color: #e6e8ee !important; }

/* ---------- Advanced answer styling ---------- */

/* Meta row above answer (reading time, etc.) */
.msg-meta {
  display: flex; gap: 14px; align-items: center;
  font-size: 11px; color: #6b7488; margin-bottom: 10px;
}
.msg-meta .pill {
  display: inline-flex; align-items: center; gap: 5px;
  background: #11141b; border: 1px solid #1f232d;
  border-radius: 999px; padding: 3px 9px;
}

/* TL;DR callout */
.tldr-card {
  background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(139,92,246,0.06) 100%);
  border: 1px solid #2e3856;
  border-left: 3px solid #8b5cf6;
  border-radius: 10px;
  padding: 12px 16px;
  margin: 4px 0 18px 0;
  font-size: 14.5px;
  line-height: 1.55;
  color: #e6e8ee;
  display: flex; gap: 10px; align-items: flex-start;
}
.tldr-card .tldr-tag {
  background: #8b5cf6; color: #fff;
  font-size: 10px; font-weight: 700;
  letter-spacing: 0.08em;
  padding: 3px 8px; border-radius: 5px;
  flex-shrink: 0; margin-top: 1px;
}

/* Section headers — make ### Answer / ### Key Points / ### Real-world Example distinct */
.msg-content h3 {
  display: inline-flex; align-items: center; gap: 6px;
  border-bottom: none; padding-bottom: 0;
  background: #1a1f2e; border: 1px solid #2a3142;
  padding: 4px 10px; border-radius: 6px;
  font-size: 11px; letter-spacing: 0.1em;
  color: #a5b4fc;
  margin: 18px 0 10px 0;
}

/* Real-world example callout */
.example-card {
  background: rgba(74,222,128,0.05);
  border: 1px solid #1f3a2e;
  border-left: 3px solid #4ade80;
  border-radius: 10px;
  padding: 10px 14px;
  margin: 6px 0 16px 0;
  font-size: 14px; line-height: 1.6;
  color: #d8dbe3;
}

/* Follow-up suggestions panel */
.followup-label {
  font-size: 11px; color: #8b9bb5;
  text-transform: uppercase; letter-spacing: 0.08em;
  font-weight: 600; margin: 8px 0 8px 0;
  display: flex; align-items: center; gap: 6px;
}
.followup-row .stButton > button {
  width: 100%;
  background: #11141b !important;
  border: 1px solid #2a3142 !important;
  color: #cfd6e4 !important;
  text-align: left !important;
  font-size: 13px !important;
  padding: 10px 12px !important;
  white-space: normal !important;
  height: auto !important;
  line-height: 1.4 !important;
  border-radius: 10px !important;
  transition: all 0.18s ease !important;
}
.followup-row .stButton > button:hover {
  background: #1a1f2e !important;
  border-color: #6366f1 !important;
  color: #fff !important;
  transform: translateY(-1px);
}

/* Action row (copy / regenerate) */
.action-row .stButton > button {
  background: transparent !important;
  border: 1px solid #232838 !important;
  color: #8b9bb5 !important;
  font-size: 12px !important;
  padding: 4px 10px !important;
  border-radius: 6px !important;
  height: auto !important;
}
.action-row .stButton > button:hover {
  background: #1a1f2e !important;
  color: #e6e8ee !important;
  border-color: #3a4256 !important;
}
</style>
"""