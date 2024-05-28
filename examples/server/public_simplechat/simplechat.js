// @ts-check
// A simple completions and chat/completions test related web front end logic
// by Humans for All

import * as du from "./datautils.mjs";
import * as ui from "./ui.mjs"

class Roles {
    static System = "system";
    static User = "user";
    static Assistant = "assistant";
}

let gBaseURL = "http://127.0.0.1:8080";

class ApiEP {
    static Type = {
        Chat: "chat",
        Completion: "completion",
    }
    static Url = {
        'chat': `${gBaseURL}/chat/completions`,
        'completion': `${gBaseURL}/completions`,
    }
}


let gUsageMsg = `
    <p class="role-system">Usage</p>
    <ul class="ul1">
    <li> Set system prompt above, to try control ai response charactersitic, if model supports same.</li>
        <ul class="ul2">
        <li> Completion mode normally wont have a system prompt.</li>
        </ul>
    <li> Enter your query to ai assistant below.</li>
        <ul class="ul2">
        <li> Completion mode doesnt insert user/role: prefix implicitly.</li>
        <li> Use shift+enter for inserting enter/newline.</li>
        </ul>
    <li> Default ContextWindow = [System, Last Query+Resp, Cur Query].</li>
        <ul class="ul2">
        <li> experiment iRecentUserMsgCnt, max_tokens, model ctxt window to expand</li>
        </ul>
    </ul>
`;


/** @typedef {{role: string, content: string}[]} ChatMessages */

class SimpleChat {

    constructor() {
        /**
         * Maintain in a form suitable for common LLM web service chat/completions' messages entry
         * @type {ChatMessages}
         */
        this.xchat = [];
        this.iLastSys = -1;
    }

    clear() {
        this.xchat = [];
        this.iLastSys = -1;
    }

    /**
     * Recent chat messages.
     * If iRecentUserMsgCnt < 0
     *   Then return the full chat history
     * Else
     *   Return chat messages from latest going back till the last/latest system prompt.
     *   While keeping track that the number of user queries/messages doesnt exceed iRecentUserMsgCnt.
     * @param {number} iRecentUserMsgCnt
     */
    recent_chat(iRecentUserMsgCnt) {
        if (iRecentUserMsgCnt < 0) {
            return this.xchat;
        }
        if (iRecentUserMsgCnt == 0) {
            console.warn("WARN:SimpleChat:SC:RecentChat:iRecentUsermsgCnt of 0 means no user message/query sent");
        }
        /** @type{ChatMessages} */
        let rchat = [];
        let sysMsg = this.get_system_latest();
        if (sysMsg.length != 0) {
            rchat.push({role: Roles.System, content: sysMsg});
        }
        let iUserCnt = 0;
        let iStart = this.xchat.length;
        for(let i=this.xchat.length-1; i > this.iLastSys; i--) {
            if (iUserCnt >= iRecentUserMsgCnt) {
                break;
            }
            let msg = this.xchat[i];
            if (msg.role == Roles.User) {
                iStart = i;
                iUserCnt += 1;
            }
        }
        for(let i = iStart; i < this.xchat.length; i++) {
            let msg = this.xchat[i];
            if (msg.role == Roles.System) {
                continue;
            }
            rchat.push({role: msg.role, content: msg.content});
        }
        return rchat;
    }

    /**
     * Add an entry into xchat
     * @param {string} role
     * @param {string|undefined|null} content
     */
    add(role, content) {
        if ((content == undefined) || (content == null) || (content == "")) {
            return false;
        }
        this.xchat.push( {role: role, content: content} );
        if (role == Roles.System) {
            this.iLastSys = this.xchat.length - 1;
        }
        return true;
    }

    /**
     * Show the contents in the specified div
     * @param {HTMLDivElement} div
     * @param {boolean} bClear
     */
    show(div, bClear=true) {
        if (bClear) {
            div.replaceChildren();
        }
        let last = undefined;
        for(const x of this.recent_chat(gMe.iRecentUserMsgCnt)) {
            let entry = ui.el_create_append_p(`${x.role}: ${x.content}`, div);
            entry.className = `role-${x.role}`;
            last = entry;
        }
        if (last !== undefined) {
            last.scrollIntoView(false);
        } else {
            if (bClear) {
                div.innerHTML = gUsageMsg;
                gMe.show_info(div);
            }
        }
    }

    /**
     * Add needed fields wrt json object to be sent wrt LLM web services completions endpoint.
     * The needed fields/options are picked from a global object.
     * Convert the json into string.
     * @param {Object} obj
     */
    request_jsonstr_extend(obj) {
        for(let k in gMe.chatRequestOptions) {
            obj[k] = gMe.chatRequestOptions[k];
        }
        return JSON.stringify(obj);
    }

    /**
     * Return a string form of json object suitable for chat/completions
     */
    request_messages_jsonstr() {
        let req = {
            messages: this.recent_chat(gMe.iRecentUserMsgCnt),
        }
        return this.request_jsonstr_extend(req);
    }

    /**
     * Return a string form of json object suitable for /completions
     * @param {boolean} bInsertStandardRolePrefix Insert "<THE_ROLE>: " as prefix wrt each role's message
     */
    request_prompt_jsonstr(bInsertStandardRolePrefix) {
        let prompt = "";
        let iCnt = 0;
        for(const chat of this.recent_chat(gMe.iRecentUserMsgCnt)) {
            iCnt += 1;
            if (iCnt > 1) {
                prompt += "\n";
            }
            if (bInsertStandardRolePrefix) {
                prompt += `${chat.role}: `;
            }
            prompt += `${chat.content}`;
        }
        let req = {
            prompt: prompt,
        }
        return this.request_jsonstr_extend(req);
    }

    /**
     * Return a string form of json object suitable for specified api endpoint.
     * @param {string} apiEP
     */
    request_jsonstr(apiEP) {
        if (apiEP == ApiEP.Type.Chat) {
            return this.request_messages_jsonstr();
        } else {
            return this.request_prompt_jsonstr(gMe.bCompletionInsertStandardRolePrefix);
        }
    }

    /**
     * Extract the ai-model/assistant's response from the http response got.
     * Optionally trim the message wrt any garbage at the end.
     * @param {any} respBody
     * @param {string} apiEP
     */
    response_extract(respBody, apiEP) {
        let theResp = {
            assistant: "",
            trimmed: "",
        }
        if (apiEP == ApiEP.Type.Chat) {
            theResp.assistant = respBody["choices"][0]["message"]["content"];
        } else {
            try {
                theResp.assistant = respBody["choices"][0]["text"];
            } catch {
                theResp.assistant = respBody["content"];
            }
        }
        if (gMe.bTrimGarbage) {
            let origMsg = theResp.assistant;
            theResp.assistant = du.trim_garbage_at_end(theResp.assistant);
            theResp.trimmed = origMsg.substring(theResp.assistant.length);
        }
        return theResp;
    }

    /**
     * Extract the ai-model/assistant's response from the http response got in streaming mode.
     * @param {any} respBody
     * @param {string} apiEP
     */
    response_extract_stream(respBody, apiEP) {
        let assistant = "";
        if (apiEP == ApiEP.Type.Chat) {
            if (respBody["choices"][0]["finish_reason"] !== "stop") {
                assistant = respBody["choices"][0]["delta"]["content"];
            }
        } else {
            try {
                assistant = respBody["choices"][0]["text"];
            } catch {
                assistant = respBody["content"];
            }
        }
        return assistant;
    }

    /**
     * Allow setting of system prompt, but only at begining.
     * @param {string} sysPrompt
     * @param {string} msgTag
     */
    add_system_begin(sysPrompt, msgTag) {
        if (this.xchat.length == 0) {
            if (sysPrompt.length > 0) {
                return this.add(Roles.System, sysPrompt);
            }
        } else {
            if (sysPrompt.length > 0) {
                if (this.xchat[0].role !== Roles.System) {
                    console.error(`ERRR:SimpleChat:SC:${msgTag}:You need to specify system prompt before any user query, ignoring...`);
                } else {
                    if (this.xchat[0].content !== sysPrompt) {
                        console.error(`ERRR:SimpleChat:SC:${msgTag}:You cant change system prompt, mid way through, ignoring...`);
                    }
                }
            }
        }
        return false;
    }

    /**
     * Allow setting of system prompt, at any time.
     * @param {string} sysPrompt
     * @param {string} msgTag
     */
    add_system_anytime(sysPrompt, msgTag) {
        if (sysPrompt.length <= 0) {
            return false;
        }

        if (this.iLastSys < 0) {
            return this.add(Roles.System, sysPrompt);
        }

        let lastSys = this.xchat[this.iLastSys].content;
        if (lastSys !== sysPrompt) {
            return this.add(Roles.System, sysPrompt);
        }
        return false;
    }

    /**
     * Retrieve the latest system prompt.
     */
    get_system_latest() {
        if (this.iLastSys == -1) {
            return "";
        }
        let sysPrompt = this.xchat[this.iLastSys].content;
        return sysPrompt;
    }

}


class MultiChatUI {

    constructor() {
        /** @type {Object<string, SimpleChat>} */
        this.simpleChats = {};
        /** @type {string} */
        this.curChatId = "";

        // the ui elements
        this.elInSystem = /** @type{HTMLInputElement} */(document.getElementById("system-in"));
        this.elDivChat = /** @type{HTMLDivElement} */(document.getElementById("chat-div"));
        this.elBtnUser = /** @type{HTMLButtonElement} */(document.getElementById("user-btn"));
        this.elInUser = /** @type{HTMLInputElement} */(document.getElementById("user-in"));
        this.elDivHeading = /** @type{HTMLSelectElement} */(document.getElementById("heading"));
        this.elDivSessions = /** @type{HTMLDivElement} */(document.getElementById("sessions-div"));
        this.elBtnSettings = /** @type{HTMLButtonElement} */(document.getElementById("settings"));

        this.validate_element(this.elInSystem, "system-in");
        this.validate_element(this.elDivChat, "chat-div");
        this.validate_element(this.elInUser, "user-in");
        this.validate_element(this.elDivHeading, "heading");
        this.validate_element(this.elDivChat, "sessions-div");
        this.validate_element(this.elBtnSettings, "settings");
    }

    /**
     * Check if the element got
     * @param {HTMLElement | null} el
     * @param {string} msgTag
     */
    validate_element(el, msgTag) {
        if (el == null) {
            throw Error(`ERRR:SimpleChat:MCUI:${msgTag} element missing in html...`);
        } else {
            console.debug(`INFO:SimpleChat:MCUI:${msgTag} Id[${el.id}] Name[${el["name"]}]`);
        }
    }

    /**
     * Reset user input ui.
     * * clear user input
     * * enable user input
     * * set focus to user input
     */
    ui_reset_userinput() {
        this.elInUser.value = "";
        this.elInUser.disabled = false;
        this.elInUser.focus();
    }

    /**
     * Setup the needed callbacks wrt UI, curChatId to defaultChatId and
     * optionally switch to specified defaultChatId.
     * @param {string} defaultChatId
     * @param {boolean} bSwitchSession
     */
    setup_ui(defaultChatId, bSwitchSession=false) {

        this.curChatId = defaultChatId;
        if (bSwitchSession) {
            this.handle_session_switch(this.curChatId);
        }

        this.elBtnSettings.addEventListener("click", (ev)=>{
            this.elDivChat.replaceChildren();
            gMe.show_settings(this.elDivChat);
        });

        this.elBtnUser.addEventListener("click", (ev)=>{
            if (this.elInUser.disabled) {
                return;
            }
            this.handle_user_submit(this.curChatId, gMe.apiEP).catch((/** @type{Error} */reason)=>{
                let msg = `ERRR:SimpleChat\nMCUI:HandleUserSubmit:${this.curChatId}\n${reason.name}:${reason.message}`;
                console.debug(msg.replace("\n", ":"));
                alert(msg);
                this.ui_reset_userinput();
            });
        });

        this.elInUser.addEventListener("keyup", (ev)=> {
            // allow user to insert enter into their message using shift+enter.
            // while just pressing enter key will lead to submitting.
            if ((ev.key === "Enter") && (!ev.shiftKey)) {
                let value = this.elInUser.value;
                this.elInUser.value = value.substring(0,value.length-1);
                this.elBtnUser.click();
                ev.preventDefault();
            }
        });

        this.elInSystem.addEventListener("keyup", (ev)=> {
            // allow user to insert enter into the system prompt using shift+enter.
            // while just pressing enter key will lead to setting the system prompt.
            if ((ev.key === "Enter") && (!ev.shiftKey)) {
                let chat = this.simpleChats[this.curChatId];
                chat.add_system_anytime(this.elInSystem.value, this.curChatId);
                chat.show(this.elDivChat);
                ev.preventDefault();
            }
        });

    }

    /**
     * Setup a new chat session and optionally switch to it.
     * @param {string} chatId
     * @param {boolean} bSwitchSession
     */
    new_chat_session(chatId, bSwitchSession=false) {
        this.simpleChats[chatId] = new SimpleChat();
        if (bSwitchSession) {
            this.handle_session_switch(chatId);
        }
    }

    /**
     * Try read json response early, if available.
     * @param {SimpleChat} chat
     * @param {Response} resp
     */
    async read_json_early(chat, resp) {
        if (!resp.body) {
            throw Error("ERRR:SimpleChat:MCUI:ReadJsonEarly:No body...");
        }
        let tdUtf8 = new TextDecoder("utf-8");
        let rr = resp.body.getReader();
        let gotBody = "";
        while(true) {
            let { value: cur,  done: done } = await rr.read();
            let curBody = tdUtf8.decode(cur);
            console.debug("DBUG:SC:PART:Str:", curBody);
            if (curBody.length > 0) {
                let curArrays = curBody.split("\n");
                for(let curArray of curArrays) {
                    console.debug("DBUG:SC:PART:StrPart:", curArray);
                    if (curArray.length <= 0) {
                        continue;
                    }
                    if (curArray.startsWith("data:")) {
                        curArray = curArray.substring(5);
                    }
                    let curJson = JSON.parse(curArray);
                    console.debug("DBUG:SC:PART:Json:", curJson);
                    gotBody += chat.response_extract_stream(curJson, gMe.apiEP);
                }
            }
            if (done) {
                break;
            }
        }
        console.debug("DBUG:SC:PART:Full:", gotBody);
        return gotBody;
    }

    /**
     * Handle user query submit request, wrt specified chat session.
     * @param {string} chatId
     * @param {string} apiEP
     */
    async handle_user_submit(chatId, apiEP) {

        let chat = this.simpleChats[chatId];

        // In completion mode, if configured, clear any previous chat history.
        // So if user wants to simulate a multi-chat based completion query,
        // they will have to enter the full thing, as a suitable multiline
        // user input/query.
        if ((apiEP == ApiEP.Type.Completion) && (gMe.bCompletionFreshChatAlways)) {
            chat.clear();
        }

        chat.add_system_anytime(this.elInSystem.value, chatId);

        let content = this.elInUser.value;
        if (!chat.add(Roles.User, content)) {
            console.debug(`WARN:SimpleChat:MCUI:${chatId}:HandleUserSubmit:Ignoring empty user input...`);
            return;
        }
        chat.show(this.elDivChat);

        let theUrl = ApiEP.Url[apiEP];
        let theBody = chat.request_jsonstr(apiEP);

        this.elInUser.value = "working...";
        this.elInUser.disabled = true;
        console.debug(`DBUG:SimpleChat:MCUI:${chatId}:HandleUserSubmit:${theUrl}:ReqBody:${theBody}`);
        let resp = await fetch(theUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: theBody,
        });

        //let respBody = await resp.json();
        let respBody = await this.read_json_early(chat, resp);
        console.debug(`DBUG:SimpleChat:MCUI:${chatId}:HandleUserSubmit:RespBody:${JSON.stringify(respBody)}`);
        let theResp = chat.response_extract(respBody, apiEP);
        chat.add(Roles.Assistant, theResp.assistant);
        if (chatId == this.curChatId) {
            chat.show(this.elDivChat);
            if (theResp.trimmed.length > 0) {
                let p = ui.el_create_append_p(`TRIMMED:${theResp.trimmed}`, this.elDivChat);
                p.className="role-trim";
            }
        } else {
            console.debug(`DBUG:SimpleChat:MCUI:HandleUserSubmit:ChatId has changed:[${chatId}] [${this.curChatId}]`);
        }
        this.ui_reset_userinput();
    }

    /**
     * Show buttons for NewChat and available chat sessions, in the passed elDiv.
     * If elDiv is undefined/null, then use this.elDivSessions.
     * Take care of highlighting the selected chat-session's btn.
     * @param {HTMLDivElement | undefined} elDiv
     */
    show_sessions(elDiv=undefined) {
        if (!elDiv) {
            elDiv = this.elDivSessions;
        }
        elDiv.replaceChildren();
        // Btn for creating new chat session
        let btnNew = ui.el_create_button("New CHAT", (ev)=> {
            if (this.elInUser.disabled) {
                console.error(`ERRR:SimpleChat:MCUI:NewChat:Current session [${this.curChatId}] awaiting response, ignoring request...`);
                alert("ERRR:SimpleChat\nMCUI:NewChat\nWait for response to pending query, before starting new chat session");
                return;
            }
            let chatId = `Chat${Object.keys(this.simpleChats).length}`;
            let chatIdGot = prompt("INFO:SimpleChat\nMCUI:NewChat\nEnter id for new chat session", chatId);
            if (!chatIdGot) {
                console.error("ERRR:SimpleChat:MCUI:NewChat:Skipping based on user request...");
                return;
            }
            this.new_chat_session(chatIdGot, true);
            this.create_session_btn(elDiv, chatIdGot);
            ui.el_children_config_class(elDiv, chatIdGot, "session-selected", "");
        });
        elDiv.appendChild(btnNew);
        // Btns for existing chat sessions
        let chatIds = Object.keys(this.simpleChats);
        for(let cid of chatIds) {
            let btn = this.create_session_btn(elDiv, cid);
            if (cid == this.curChatId) {
                btn.className = "session-selected";
            }
        }
    }

    create_session_btn(elDiv, cid) {
        let btn = ui.el_create_button(cid, (ev)=>{
            let target = /** @type{HTMLButtonElement} */(ev.target);
            console.debug(`DBUG:SimpleChat:MCUI:SessionClick:${target.id}`);
            if (this.elInUser.disabled) {
                console.error(`ERRR:SimpleChat:MCUI:SessionClick:${target.id}:Current session [${this.curChatId}] awaiting response, ignoring switch...`);
                alert("ERRR:SimpleChat\nMCUI:SessionClick\nWait for response to pending query, before switching");
                return;
            }
            this.handle_session_switch(target.id);
            ui.el_children_config_class(elDiv, target.id, "session-selected", "");
        });
        elDiv.appendChild(btn);
        return btn;
    }

    /**
     * Switch ui to the specified chatId and set curChatId to same.
     * @param {string} chatId
     */
    async handle_session_switch(chatId) {
        let chat = this.simpleChats[chatId];
        if (chat == undefined) {
            console.error(`ERRR:SimpleChat:MCUI:HandleSessionSwitch:${chatId} missing...`);
            return;
        }
        this.elInSystem.value = chat.get_system_latest();
        this.elInUser.value = "";
        chat.show(this.elDivChat);
        this.elInUser.focus();
        this.curChatId = chatId;
        console.log(`INFO:SimpleChat:MCUI:HandleSessionSwitch:${chatId} entered...`);
    }

}


class Me {

    constructor() {
        this.defaultChatIds = [ "Default", "Other" ];
        this.multiChat = new MultiChatUI();
        this.bCompletionFreshChatAlways = true;
        this.bCompletionInsertStandardRolePrefix = false;
        this.bTrimGarbage = true;
        this.iRecentUserMsgCnt = 2;
        this.sRecentUserMsgCnt = {
            "Full": -1,
            "Last0": 1,
            "Last1": 2,
            "Last2": 3,
            "Last4": 5,
        };
        this.apiEP = ApiEP.Type.Chat;
        // Add needed fields wrt json object to be sent wrt LLM web services completions endpoint.
        this.chatRequestOptions = {
            "temperature": 0.7,
            "max_tokens": 1024,
            "n_predict": 1024,
            //"frequency_penalty": 1.2,
            //"presence_penalty": 1.2,
        };
    }

    /**
     * Show the configurable parameters info in the passed Div element.
     * @param {HTMLDivElement} elDiv
     */
    show_info(elDiv) {

        let p = ui.el_create_append_p("Settings (devel-tools-console document[gMe])", elDiv);
        p.className = "role-system";

        ui.el_create_append_p(`bCompletionFreshChatAlways:${this.bCompletionFreshChatAlways}`, elDiv);

        ui.el_create_append_p(`bCompletionInsertStandardRolePrefix:${this.bCompletionInsertStandardRolePrefix}`, elDiv);

        ui.el_create_append_p(`bTrimGarbage:${this.bTrimGarbage}`, elDiv);

        ui.el_create_append_p(`iRecentUserMsgCnt:${this.iRecentUserMsgCnt}`, elDiv);

        ui.el_create_append_p(`chatRequestOptions:${JSON.stringify(this.chatRequestOptions)}`, elDiv);

        ui.el_create_append_p(`ApiEndPoint:${this.apiEP}`, elDiv);

    }

    /**
     * Show settings ui for configurable parameters, in the passed Div element.
     * @param {HTMLDivElement} elDiv
     */
    show_settings(elDiv) {

        let bb = ui.el_creatediv_boolbutton("SetCompletionFreshChatAlways", "CompletionFreshChatAlways", {true: "[+] yes fresh", false: "[-] no, with history"}, this.bCompletionFreshChatAlways, (val)=>{
            this.bCompletionFreshChatAlways = val;
        });
        elDiv.appendChild(bb);

        bb = ui.el_creatediv_boolbutton("SetCompletionInsertStandardRolePrefix", "CompletionInsertStandardRolePrefix", {true: "[+] yes insert", false: "[-] dont insert"}, this.bCompletionInsertStandardRolePrefix, (val)=>{
            this.bCompletionInsertStandardRolePrefix = val;
        });
        elDiv.appendChild(bb);

        bb = ui.el_creatediv_boolbutton("SetTrimGarbage", "TrimGarbage", {true: "[+] yes trim", false: "[-] dont trim"}, this.bTrimGarbage, (val)=>{
            this.bTrimGarbage = val;
        });
        elDiv.appendChild(bb);

        let sel = ui.el_creatediv_select("SetChatHistoryInCtxt", "ChatHistoryInCtxt", this.sRecentUserMsgCnt, this.iRecentUserMsgCnt, (val)=>{
            this.iRecentUserMsgCnt = this.sRecentUserMsgCnt[val];
        });
        elDiv.appendChild(sel);

        sel = ui.el_creatediv_select("SetApiEP", "ApiEndPoint", ApiEP.Type, this.apiEP, (val)=>{
            this.apiEP = ApiEP.Type[val];
        });
        elDiv.appendChild(sel);

    }

}


/** @type {Me} */
let gMe;

function startme() {
    console.log("INFO:SimpleChat:StartMe:Starting...");
    gMe = new Me();
    document["gMe"] = gMe;
    document["du"] = du;
    for (let cid of gMe.defaultChatIds) {
        gMe.multiChat.new_chat_session(cid);
    }
    gMe.multiChat.setup_ui(gMe.defaultChatIds[0], true);
    gMe.multiChat.show_sessions();
}

document.addEventListener("DOMContentLoaded", startme);
