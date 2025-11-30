Generate for me EXACTLY 50 synthetic users in JSON format. For the USER IDUse the pattern:

Each user must follow this exact schema:

{
  "user_id": "synthetic-user-001",
  "first_name": "Lorenzo",
  "last_name": "Rossi",
  "age": 21,
  "country": "Italy",
  "languages": [
    "Italian",
    "English"
  ],
  "current_status": "University student",
  "field_of_study": "Computer Engineering",
  "job_role": "Software engineer",
  "financial_support_per_year": 25000,
  "financial_support_duration": 2,
  "description": "I’m a 21-year-old computer engineering student from Italy, currently in my final year. My focus is entirely on backend systems and distributed architectures. What motivates me is the challenge of building things that don’t just work, but work at scale and never fail. I’m the kind of person who reads database documentation for fun and will spend a weekend benchmarking different caching strategies. Behaviourally, I’m extremely structured and a bit obsessive about code quality. I live by my task lists and use Notion to plan my study sprints. I struggle in chaotic, last-minute environments; I need clear goals and measurable outcomes. I’m not the fastest at creative brainstorming, but once a direction is set, I am incredibly reliable. I almost never miss a deadline. My long-term goal is to work on high-reliability infrastructure, like in fintech or cloud services. To do this, I need a Master's degree in Distributed Systems. I'm aiming for programs at top-tier universities like ETH Zurich or EPFL. The problem is purely financial. My family runs a small restaurant, and while they support me, they have no savings for this. The 25,000 EUR per year I'm requesting would cover the tuition and high cost of living, allowing me to pursue an education that is simply impossible for me to access otherwise.",
  "financial_support_return": "Income_Share"
}

CONSTRAINTS ON FIELDS

- user_id:
  - String.
  - Must be unique for each user.

- first_name, last_name:
  - Realistic names consistent with the country and languages.

- age:
  - Integer, generally between 16 and 40.
  - Must be coherent with their status and story.

- financial_support_duration:
  - Integer number of years needed (1–5), matching the study plan in the description.

- description:
  - VERY IMPORTANT: length must be around 200–500 words.
  - Must be written in natural, first-person voice.
  - Must clearly explain:
    - Their background (studies, work experience, skills, interests).
    - Their current situation (what they are studying or working on now).
    - Their concrete higher-education goal (e.g. specific degree, master’s, country or type of program).
    - The financial barrier (family context, income, local costs, why they cannot self-fund).
    - How the requested financial support (amount per year and duration) would unlock their goal.
  - The description must be coherent with all the structured fields (age, field_of_study, status, job_role, amount, duration, etc.).
  - Style: similar level of detail and realism as the examples below, including behavioural traits, work style, and motivation.

GLOBAL STORY CONSTRAINT

- All the users you create are:
  - Students and/or workers who want to pursue higher education (university degree, master’s, specialized program).
  - They do NOT have the financial possibility to do so without help.
  - The financial support they ask for is the main blocker between them and their goal.
- Ensure diversity across:
  - Countries, socio-economic backgrounds, ages, genders.
  - Fields of study, current_status, job_role.
  - Types of programs (bachelor, master, PhD, professional programs, part-time vs full-time, domestic vs abroad).

STYLE EXAMPLES

Use these texts as style and quality references. Match:
- Level of detail.
- Balance between background, behaviour, and financial context.
- Tone (honest, specific, non-generic).

[Insert here the two example paragraphs you already have about the computer engineering student and the psychology student.]