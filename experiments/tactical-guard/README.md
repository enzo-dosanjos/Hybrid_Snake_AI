# Twenty-turn tactical guard

The proof horizon includes the candidate move and up to 19 subsequent player turns. A player can force a win if at least one move wins against every opponent reply. Exhausting the horizon leaves the move unresolved. The search applies competition collision and growth rules on a two-player board with at most 256 cells including its padded border.

The guard keeps an immediate Minimax terminal decision or sole legal move. Otherwise it prefers proven wins, then unresolved moves. It keeps the original selection when eligible or when all moves lose. Otherwise it takes the highest Minimax score among eligible moves. The hybrid's original selection uses the frozen checkpoint within the 1.0-point margin.

From the project root build with `make -f experiments/tactical-guard/Makefile build/experiments/Evaluate`. The evaluator reads one competition setup on stdin and prints the win flag for the configured seat and total scheduled turns including the trapped turn. Supply the frozen ranked checkpoint as the optional argument for the hybrid.

The tactical20 table compares single exceptions. The unresolved table compares repeated guarding throughout play. The override audit records individual alternatives and retrospective 24-turn proof labels. Compare policies within their own table because the layout sets differ.
