# Maya frontend

This React/Vite client connects to the local FastAPI call-start endpoint and
joins the LiveKit room returned after the voice agent is ready.

~~~sh
npm ci
npm run dev
~~~

Useful checks:

~~~sh
npm run lint
npm run build
~~~

The API URL is currently configured in src/App.jsx for local development at
http://localhost:8000.
