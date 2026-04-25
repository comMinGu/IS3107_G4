import express from "express";
import cors from "cors";
import pg from "pg";

const { Pool } = pg;

const app = express();

const pool = new Pool({
  host: process.env.PGHOST,
  port: Number(process.env.PGPORT || 5432),
  database: process.env.PGDATABASE,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  ssl: process.env.PGSSLMODE === "disable" ? false : { rejectUnauthorized: false },
  max: 5,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
});

app.use(cors({
  origin: "*",
  methods: ["GET"],
}));

app.get("/health", async (req, res) => {
  try {
    await pool.query("SELECT 1");
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get("/prediction", async (req, res) => {
  const gameId = req.query.game_id;
  const turnRaw = req.query.turn;
  const sideToMove = req.query.side_to_move;

  if (!gameId || !turnRaw || !sideToMove) {
    return res.status(400).json({
      error: "Missing required query params: game_id, turn, side_to_move"
    });
  }

  const turn = Number(turnRaw);
  if (!Number.isFinite(turn)) {
    return res.status(400).json({ error: "Invalid turn" });
  }

  if (!["w", "b"].includes(sideToMove)) {
    return res.status(400).json({ error: "side_to_move must be 'w' or 'b'" });
  }

  try {
    const sql = `
      SELECT probabilities, predicted_label
      FROM public.lichess_move_predictions
      WHERE game_id = $1
        AND turn = $2
        AND side_to_move = $3
      LIMIT 1
    `;

    const result = await pool.query(sql, [gameId, turn, sideToMove]);

    if (!result.rows.length) {
      return res.status(404).json({ error: "No prediction row found" });
    }

    const row = result.rows[0];

    res.json({
      probabilities: row.probabilities,
      predicted_label: row.predicted_label ?? null
    });
  } catch (err) {
    res.status(500).json({
      error: "Database query failed",
      details: err.message
    });
  }
});

const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Prediction API listening on port ${port}`);
});