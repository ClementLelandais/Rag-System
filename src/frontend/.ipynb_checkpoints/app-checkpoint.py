"""
Module Frontend Application
===============

Ce module fournit l'application frontend utilisant Gradio pour interagir
avec le système RAG (Retrieval-Augmented Generation) via l'API backend.
"""

import gradio as gr
from src.frontend.api_client import APIClient

api_client = APIClient()

# ---------- Réponse ----------
def respond(message: str) -> str:
    try:
        return api_client.query(message)
    except Exception as e:
        return f"Erreur: {e}"

# ---------- État et helpers ----------
def _init_state():
    return {"conversations": {"Chat 1": []}, "current": "Chat 1", "next_idx": 2}

def _choices(state):
    return list(state["conversations"].keys())

# Nouveau chat
def new_chat(state):
    name = f"Chat {state['next_idx']}"
    state["conversations"][name] = []
    state["current"] = name
    state["next_idx"] += 1
    return (
        state,
        gr.update(value=name, choices=_choices(state)),
        state["conversations"][name],
    )

# Envoyer un message
def send_message(user_message, state):
    if not user_message.strip():
        return "", state["conversations"][state["current"]], state

    cur = state["current"]
    reply = respond(user_message)
    state["conversations"][cur].append({"role": "user", "content": user_message})
    state["conversations"][cur].append({"role": "assistant", "content": reply})
    return "", state["conversations"][cur], state

# Charger un chat
def load_chat(selected, state):
    if selected and selected in state["conversations"]:
        state["current"] = selected
    return state, state["conversations"][state["current"]]

# Vider le chat courant
def clear_current(state):
    state["conversations"][state["current"]] = []
    return state, []

# Supprimer le chat courant
def delete_current(state):
    cur = state["current"]
    if cur in state["conversations"]:
        del state["conversations"][cur]
    if not state["conversations"]:
        state = _init_state()
    else:
        state["current"] = _choices(state)[0]
    return (
        state,
        gr.update(value=state["current"], choices=_choices(state)),
        state["conversations"][state["current"]],
    )

# ---------- UI ----------
with gr.Blocks(title="Akelio - RAG System", fill_height=True) as app:
    initial_state = _init_state()
    st = gr.State(initial_state)

    gr.Markdown("## Akelio - RAG System")

    # Barre latérale type ChatGPT
    with gr.Sidebar():
        gr.Markdown("### Conversations")
        btn_new = gr.Button("Nouveau chat", variant="primary")
        chat_list = gr.Radio(
            label="Historique",
            choices=_choices(initial_state),
            value="Chat 1",
            interactive=True,
        )
        gr.Markdown("---")
        btn_clear  = gr.Button("Vider ce chat",    variant="secondary")
        btn_delete = gr.Button("Supprimer ce chat", variant="stop")

    # Zone principale
    chatbot = gr.Chatbot(
        label="Chatbot",
        height=500,
        value=[],
        type="messages",   
        show_copy_button=True,
    )
    msg = gr.Textbox(
        placeholder="Pose ta question…",
        show_label=False,
        autofocus=True,
    )

    # Interactions
    msg.submit(send_message, [msg, st], [msg, chatbot, st])
    btn_new.click(new_chat, st, [st, chat_list, chatbot])
    chat_list.change(load_chat, [chat_list, st],  [st, chatbot])
    btn_clear.click(clear_current, st, [st, chatbot])
    btn_delete.click(delete_current, st, [st, chat_list, chatbot])

if __name__ == "__main__":
    app.launch(server_port=7860, server_name="0.0.0.0")