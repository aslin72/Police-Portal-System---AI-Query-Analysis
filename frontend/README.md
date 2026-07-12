# Police Portal Frontend

The frontend is a Next.js App Router application for citizen complaint intake and officer triage workflows.

## Development

Install dependencies and start the development server:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The FastAPI backend must be running separately on `http://localhost:8000`.

To use another backend URL, create `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_URL=https://api.example.gov.in
```

## Commands

```bash
npm run dev
npm run lint
npm run build
npm run start
```

Primary routes live under `src/app/citizen` and `src/app/officer`. Shared API types and requests are defined in `src/lib/types.ts`.
