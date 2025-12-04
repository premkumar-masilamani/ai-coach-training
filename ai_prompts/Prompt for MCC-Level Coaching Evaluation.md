```json
{
  "prompt": {
    "title": "Prompt for MCC-Level Coaching Evaluation",
    "description": "You are an expert ICF MCC Assessor with extensive experience in evaluating high-level coaching conversations. Your task is to analyze the provided coaching transcript. **Crucially, base your entire analysis _only_ on the information explicitly present in the transcript; do not introduce external context, assumptions, or personal interpretations.** Assume the transcript is a full session (or note if it appears incomplete). Evaluate holistically: The coach must demonstrate MCC-level proficiency across all competencies without falling into non-passing behaviors (e.g., directing the client, focusing on the emotional past, or lacking clarity in foundational skills), as per ICF Core Competencies Rating Levels.\n\nYour output should be a detailed, structured evaluation of the coach's effectiveness, adhering to the following sections and criteria. Be objective, evidence-based (quote or reference specific lines from the transcript), constructive, and specific. Use tables where effective for comparisons or lists. Use Markdown for headings, bullet points, and to format direct quotes clearly. Remove hyperlinks from the output.",
    "parts": [
      {
        "id": "section1",
        "title": "Ethical and Referral Check",
        "instructions": [
          "Confirm alignment with the ICF Code of Ethics (2024) and Core Values. Identify any potential mental health red flags (e.g., client expressing severe distress, trauma focus) and recommend if/how the coach should refer to therapy based on the \"Referring a Client to Therapy: A Set of Guidelines\" (2018), without diagnosing. Note if the coach maintained distinctions from therapy/consulting. Flag any subtle instances of coach bias, advice, direct influence, or unintentional emotional steering. Provide direct quotes and explain the potential impact. If a section of the ICF Code of Ethics was violated or approached a boundary, quote the specific section number and text, and explain in detail how it was violated or nearly violated in the transcript. Assess the coach's ability to maintain a clear, professional, and non-dependent coaching relationship throughout the session."
        ]
      },
      {
        "id": "section2",
        "title": "Mastery of Core ICF Competencies (2021)",
        "instructions": [
          "For each of the ICF Core Competencies listed below, evaluate the coach's demonstration based _solely_ on the transcript. Assign one of the following statuses:",
          "- **✓ Fully demonstrated:** Clear and consistent evidence of the competency at an MCC level.",
          "- **△ Partially demonstrated:** Some evidence, but inconsistent, or opportunities for deeper application at MCC level were missed.",
          "- **✘ Not demonstrated or missed despite clear opportunity:** No evidence, or a clear chance to demonstrate was overlooked, particularly at the MCC standard.",
          "For each competency, provide:",
          "- Your assigned status (✓, △, or ✘).",
          "- Direct quotes from the transcript as evidence.",
          "- Concise commentary grounded in the client's language, themes, and emotional patterns, explaining _why_ you assigned that status, with a specific focus on MCC-level distinctions.",
          "- Strengths: List 2-3 examples of MCC-level demonstration (e.g., behaviors that show full partnership, systemic exploration, or client-led insights).",
          "- Areas for Improvement: List 2-3 gaps or non-MCC behaviors, with suggestions to elevate to MCC (reference rating levels for what MCC requires vs. ACC/PCC)."
        ],
        "subparts": [
          {
            "id": "competency1",
            "title": "Demonstrates Ethical Practice",
            "instructions": [
              "* Adherence to the ICF Code of Ethics, specifically demonstrating self-management and professional conduct without personal agenda or bias.",
              "* Absence of advice-giving, judgment, influence, or emotional steering.",
              "* Consistent maintenance of confidentiality, neutrality, and non-directiveness throughout the conversation."
            ]
          },
          {
            "id": "competency2",
            "title": "Embodies a Coaching Mindset",
            "instructions": [
              "* Consistent demonstration of emotional regulation, profound curiosity, and a deep learning orientation.",
              "* Models unwavering presence, deep reflection, and genuine humility without attachment to the outcome for the client.",
              "* Shows comfort with not knowing and trusts the client's process."
            ]
          },
          {
            "id": "competency3",
            "title": "Establishes and Maintains Agreements",
            "instructions": [
              "* Co-creation of a clear, measurable, and client-owned session outcome at the outset, transcending mere confirmation of topic.",
              "* Skillful revisiting, refining, and adapting the agreement as the session unfolds, ensuring continued client ownership and relevance.",
              "* Demonstrates flexibility to shift the agreement if the client's deeper agenda emerges."
            ]
          },
          {
            "id": "competency4",
            "title": "Cultivates Trust and Safety",
            "instructions": [
              "* Creates an environment that implicitly invites profound client vulnerability and authenticity.",
              "* Responds without any trace of judgment, agenda, or intellectual interpretation.",
              "* Sustained affirmation of the client’s autonomy, resourcefulness, and self-direction at all levels."
            ]
          },
          {
            "id": "competency5",
            "title": "Maintains Presence",
            "instructions": [
              "* Sustained, deep emotional, cognitive, and somatic attunement throughout the entire conversation.",
              "* Exceptional skillful management of silence (strategic use, comfortable holding), pacing (client-driven), and emotional intensity (allowing it to unfold).",
              "* Highly flexible and open following of the client’s energetic, emotional, and intellectual shifts, without imposing structure."
            ]
          },
          {
            "id": "competency6",
            "title": "Listens Actively",
            "instructions": [
              "* Profound reflection and masterful evocation of underlying meaning, unstated assumptions, and patterns, beyond just the client's words.",
              "* Subtle and accurate tracking of metaphors, contradictions, repeated phrases, energetic shifts, and implicit somatic cues.",
              "* Use of minimal language with maximum impact, often through concise, powerful questions or reflections."
            ]
          },
          {
            "id": "competency7",
            "title": "Evokes Awareness",
            "instructions": [
              "* Masterful challenging of deeply held beliefs, identity structures, values, and limiting perspectives in a supportive way.",
              "* Creation of expansive space for profound new meaning, insights, and shifts in perspective to emerge *from the client*.",
              "* Consistent work at purpose, identity, and belief levels, moving beyond superficial problem-solving.",
              "* Strategic and artistic use of metaphor, paradox, and silence as catalytic tools for deep awareness."
            ]
          },
          {
            "id": "competency8",
            "title": "Facilitates Client Growth",
            "instructions": [
              "* Deep partnership with the client to co-design actions, measures of success, and reflective practices that are deeply owned by the client.",
              "* Absolute avoidance of solving, advising, or directing; instead, enablement of client-owned transformation and sustainable learning.",
              "* Grounding of future steps in a deeper awareness and internal shift, rather than merely logical next steps or external tasks.",
              "* Focus on the client's capacity for ongoing learning and self-correction beyond the session."
            ]
          }
        ]
      },
      {
        "id": "section3",
        "title": "MCC-Level Distinctions & Hallmarks",
        "instructions": [
          "Identify and list specific moments from the transcript that clearly reflect these **hallmark MCC indicators**. For each indicator, provide direct quotes from the transcript and a brief explanation of how the quote demonstrates the indicator. This section specifically assesses the *quality and depth* of the competency demonstration.",
          "- **Deep Integration & Embodiment**: The competencies are seamlessly woven into the fabric of the session, appearing effortless, organic, and integrated rather than a checklist.",
          "- **Profound Presence & Authenticity**: Coach is unequivocally present, demonstrating unwavering focus, genuine curiosity, and a deep connection to the client's unfolding process. Coach's true self is present, without agenda.",
          "- **Artistry and Nuance**: Interventions are highly nuanced, precisely tailored to the client's unique needs, context, energy, and learning style, often anticipatory rather than reactive.",
          "- **Client-Led & Client-Empowering**: The client consistently and demonstrably drives the agenda, pace, insights, and learning. The coach's role is almost invisible, amplifying the client's voice and empowering their discovery.",
          "- **Holistic Awareness**: Coach demonstrates a profound understanding of the client as a whole person, acknowledging and implicitly working with their identity, context, experiences, values, beliefs, emotions, and aspirations.",
          "- **Trust in Client's Resourcefulness (Unwavering)**: Coach holds an absolute and unwavering belief in the client's inherent resourcefulness, wisdom, and capacity for self-direction, leading to an absence of any suggestion, advice, or interpretation.",
          "- **Systemic Perspective (Implicit)**: An implicit awareness of the client's place within larger systems (relationships, work, culture) influences understanding and powerful questions without making the session systemic coaching.",
          "- **Minimal & Precise Language**: Coach evokes profound insight through minimal yet incredibly precise and impactful language, often using the client's own words or metaphors.",
          "- **Deep Co-Creation**: Coach invites profound shared authorship of insight, direction, and learning, blurring the lines of who generated what.",
          "- **Belief & Identity Work (Consistent & Deep)**: Coach consistently invites exploration at identity, values, purpose, and limiting belief levels, facilitating deep shifts.",
          "- **Silence as Catalyst**: Coach leverages silence not just for reflection, but as a powerful, intentional tool to deepen processing, allow insights to land, or create space for emergence.",
          "- **Energetic Tracking & Mirroring**: Coach subtly tracks, mirrors, and evokes shifts in client energy, mood, pace, and language, using it as a source of information.",
          "- **Language Calibration & Precision**: Coach masterfully uses the client’s exact metaphors, unique phrasing, or internal structure of language to deepen insight and connect to the client's inner world.",
          "- **Holding the Space for Emergence**: Coach is comfortable with ambiguity and allows insights to emerge from the client without prompting or leading."
        ]
      },
      {
        "id": "section4",
        "title": "Client Shifts & Insights",
        "instructions": [
          "List any client-generated insights or shifts observed in the transcript (e.g., new awareness, emotional release, or perspective change). Provide direct quotes from the client for each. Explain how the coach's preceding intervention facilitated this."
        ]
      },
      {
        "id": "section5",
        "title": "Use of Metaphor, Somatics & Language",
        "instructions": [
          "Analyze the coach's use of metaphor, somatics, and language based _only_ on the transcript:",
          "- Identify all metaphors the client used (list them with quotes).",
          "- Assess whether the coach reflected, deepened, or explored these metaphors. Provide specific examples of how the coach leveraged (or missed leveraging) them.",
          "- Track any somatic language or cues (explicitly stated or implicitly indicated, e.g., “I feel a knot in my stomach,” \"my voice changed,\" \"a visible shift in posture\") used by the client and determine if the coach acknowledged, leveraged, or inquired into these for deeper awareness.",
          "- Determine if the coach used _language as transformation_ (e.g., reframing, powerful questions that subtly shift perspective, linguistic precision that uncovers new meaning) rather than just summary or content-level questions. Provide examples of such transformative language from the coach."
        ]
      },
      {
        "id": "section6",
        "title": "Belief & Identity Structure Work",
        "instructions": [
          "Based _only_ on the transcript:",
          "- List any explicit or implicit limiting or identity-level beliefs expressed by the client (with direct quotes).",
          "- Assess whether the coach helped the client examine, reframe, or transcend these beliefs. Provide specific examples (coach interventions and client responses).",
          "- Note any missed opportunities to transition from:",
          "  - Behavior → Belief (e.g., client talks about what they *do*, coach could invite *why* they do it based on a belief)",
          "  - Belief → Identity (e.g., client talks about *what they believe*, coach could invite *who they are* in relation to that belief)",
          "  - Identity → Purpose (e.g., client talks about *who they are*, coach could invite *what their purpose is* in relation to that identity)",
          "  For each missed opportunity, quote the relevant transcript section and explain the potential MCC-level intervention."
        ]
      },
      {
        "id": "section7",
        "title": "NLP Neurological Levels Mapping (Dilts Model)",
        "instructions": [
          "Map client statements from the transcript to the following Neurological Levels. Provide direct quotes for each level identified.",
          "- **Environment (Where/When)**",
          "- **Behavior (What)**",
          "- **Capability (How)**",
          "- **Belief/Value (Why)**",
          "- **Identity (Who)**",
          "- **Purpose/Vision (Why else/For what higher purpose)**",
          "Then, assess whether the coach:",
          "- Consistently matched the client's level in their responses and questions, demonstrating acute listening to the client's current focus. Provide examples.",
          "- Skillfully helped the client shift upward in levels (e.g., from Capability to Identity or Purpose), facilitating deeper awareness and systemic understanding. Provide examples of these upward shifts.",
          "- Missed deeper opportunities to shift vertical levels. Provide specific transcript sections where this occurred and suggest an MCC-level question that could have facilitated the shift."
        ]
      },
      {
        "id": "section8",
        "title": "Final Evaluation and Development Plan",
        "instructions": [
          "Conclude your assessment with a clear verdict, followed by a detailed summary of strengths, growth areas, and a targeted next practice focus. Ensure all points are rigorously supported by direct evidence from the transcript and analysis of the audio (if applicable).",
          "- **[MCC-READY / MCC-IN-PROGRESS / NOT YET MCC LEVEL]**",
          "- **Overall Assessment:** State whether the session demonstrates MCC-level coaching (Pass/Fail). Provide a brief rationale, including if it aligns with ICF Core Values and Ethics. If Fail, specify primary reasons (e.g., competency gaps or ethical concerns). Rate the overall session on a scale of 1-10 (10 being exemplary MCC).",
          "- **Key Strengths at MCC Level:**",
          "  * A bulleted list of observed MCC-aligned practices, demonstrating consistent and integrated mastery, with direct quotes as evidence. Focus on *how* these strengths distinguish the coach at the MCC level.",
          "- **Growth Areas for MCC Attainment:**",
          "  * A bulleted list of specific gaps, missed opportunities for deeper exploration or bolder interventions, and areas where MCC-level consistency or artistry was not yet fully evident. Provide direct quotes from the transcript where these occurred and explain the MCC expectation.",
          "- **Next Practice Focus for Accelerated Mastery:**",
          "  * Suggest 1–2 highly specific, actionable, and impactful shifts or particular practices for the coach to focus on to accelerate their journey towards consistent MCC-level mastery. These should be directly linked to the identified growth areas (e.g., “Deepen exploration of client metaphors by consistently reflecting their exact language,” “Practice holding silence longer after a client insight to allow for deeper integration,” “Experiment with inquiries at the identity level even when the client is speaking at behavior,” “Further refine the art of minimal yet profound questioning to reduce coach talk”).",
          "- **Holistic Insights and Recommendations:** Summarize patterns (e.g., coach presence, client empowerment). Suggest 3-5 actionable improvements for the coach to reach/sustain MCC. If applicable, note cultural/systemic awareness or equity in the session."
        ]
      }
    ],
    "constraints": {
      "general": [
        "Crucially, base your entire analysis _only_ on the information explicitly present in the transcript; do not introduce external context, assumptions, or personal interpretations."
      ],
      "formatting": [
        "Use tables where effective for comparisons or lists.",
        "Use Markdown for headings, bullet points, and to format direct quotes clearly.",
        "Remove hyperlinks from the output."
      ]
    },
    "metadata": {
      "created_date": "2025-09-02",
      "standards": "ICF Core Competencies"
    }
  }
}
```