import math

from model import (
    Location,
    Portal,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import ReasoningWizard
from dataclasses import dataclass

# WizardGreedy: Calculates the heuristic of closest goblin minus portal, with the terminal
# portal goal and defeat goblins making the greedy agent more decisive. Using larger values
# make the portal state overwhelmingly better than the dead states. This allows the wizard
# to move less and finish more quickly.
class WizardGreedy(ReasoningWizard):
    def evaluation(self, state: GameState) -> float:
        # if Wizard dead, return -100 (really bad score, should avoid)
        if len(state.get_all_entity_locations(Wizard)) == 0:
            return -100

        wiz = state.get_all_entity_locations(Wizard)[0]
        portal = state.get_all_tile_locations(Portal)[0]

        # if Wizard gets to portal, return 300 (really GOOD score, prefer!)
        if wiz == portal:
            return 300


        p_dist = abs(wiz.col - portal.col) + abs(wiz.row - portal.row)

        goblins = state.get_all_entity_locations(Goblin)

        # initially set closest_g_dist to a big number
        closest_g_dist = 100000
        for goblin in goblins:
            g_dist = abs(wiz.col - goblin.col) + abs(wiz.row - goblin.row)
            # if this goblin closer to the wizard than previous goblin, update
            if g_dist < closest_g_dist:
                closest_g_dist = g_dist

        # heuristic = manhattan(closest goblin - wizard) - manhattan(wizard - portal)
        return closest_g_dist - p_dist

# WizardMiniMax: Calculates best path using maximizer (Wizard) and minimizer (Goblin) nodes
class WizardMiniMax(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # if Wizard dead, return -100 (really bad score, should avoid)
        if len(state.get_all_entity_locations(Wizard)) == 0:
            return -100

        wiz = state.get_all_entity_locations(Wizard)[0]
        portal = state.get_all_tile_locations(Portal)[0]

        # if Wizard gets to portal, return 300 (really GOOD score, prefer!)
        if wiz == portal:
            return 300

        p_dist = abs(wiz.col - portal.col) + abs(wiz.row - portal.row)
        goblins = state.get_all_entity_locations(Goblin)

        # initially set closest_g_dist to a big number
        closest_g_dist = 100000
        for goblin in goblins:
            g_dist = abs(wiz.col - goblin.col) + abs(wiz.row - goblin.row)
            # if this goblin closer to the wizard than previous goblin, update
            if g_dist < closest_g_dist:
                closest_g_dist = g_dist

        # heuristic = manhattan(closest goblin - wizard) - manhattan(wizard - portal)
        return closest_g_dist - p_dist

    def is_terminal(self, state: GameState) -> bool:
        return (len(state.get_all_entity_locations(Wizard)) == 0 or
                (state.get_all_entity_locations(Wizard)[0] ==
                 state.get_all_tile_locations(Portal)[0]))

    def react(self, state: GameState) -> WizardMoves:
        # edited ReasoningWizard's react to fit MiniMax algorithm
        values: dict[WizardMoves, float] = {}
        successors = self.get_successors(state)

        for (action, succ_state) in successors:
            values[action] = self.minimax(succ_state, self.max_depth - 1)

        return max(values, key=values.get)


    def minimax(self, state: GameState, depth: int):
        if (self.is_terminal(state) or depth == 0):
            return self.evaluation(state)
        successors = self.get_successors(state)
        current_ent = state.get_active_entity()
        ret = None


        if isinstance(current_ent, Wizard):
            # Wizard Maximizer - starts comparing to -infinity
            ret = -(math.inf)
            for (action, succ_state) in successors:
                ret = max(ret, self.minimax(succ_state, depth-1))
        elif isinstance(current_ent, Goblin):
            # Goblin Minimizer - starts comparing to infinity
            ret = math.inf
            for (action, succ_state) in successors:
                ret = min(ret, self.minimax(succ_state, depth - 1))

        return ret

# WizardAlphaBeta: Calculates same as MiniMax but prunes branches that will never be needed
class WizardAlphaBeta(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # if Wizard dead, return -100 (really bad score, should avoid)
        if len(state.get_all_entity_locations(Wizard)) == 0:
            return -100

        wiz = state.get_all_entity_locations(Wizard)[0]
        portal = state.get_all_tile_locations(Portal)[0]

        # if Wizard gets to portal, return 300 (really GOOD score, prefer!)
        if wiz == portal:
            return 300

        p_dist = abs(wiz.col - portal.col) + abs(wiz.row - portal.row)
        goblins = state.get_all_entity_locations(Goblin)

        # initially set closest_g_dist to a big number
        closest_g_dist = 100000
        for goblin in goblins:
            g_dist = abs(wiz.col - goblin.col) + abs(wiz.row - goblin.row)
            # if this goblin closer to the wizard than previous goblin, update
            if g_dist < closest_g_dist:
                closest_g_dist = g_dist

        # heuristic = manhattan(closest goblin - wizard) - manhattan(wizard - portal)
        return closest_g_dist - p_dist

    def is_terminal(self, state: GameState) -> bool:
        return (len(state.get_all_entity_locations(Wizard)) == 0 or
                (state.get_all_entity_locations(Wizard)[0] ==
                 state.get_all_tile_locations(Portal)[0]))

    def react(self, state: GameState) -> WizardMoves:
        alpha = -math.inf
        beta = math.inf

        values: dict[WizardMoves, float] = {}
        successors = self.get_successors(state)
        successors = sorted(successors, key=lambda pair: self.evaluation(pair[1]), reverse=True)

        for (action, succ_state) in successors:
            values[action] = self.alpha_beta_minimax(succ_state, self.max_depth - 1, alpha, beta)
            alpha = max(alpha, values[action])

        return max(values, key=values.get)



    def alpha_beta_minimax(self, state: GameState, depth: int, alpha: float, beta: float):
        if self.is_terminal(state) or depth == 0:
            return self.evaluation(state)

        successors = self.get_successors(state)
        current_ent = state.get_active_entity()
        val = None

        if isinstance(current_ent, Wizard):
            # Wizard Maximizer - starts comparing to -infinity
            val = -(math.inf)
            successors = sorted(successors, key=lambda pair: self.evaluation(pair[1]), reverse=True)

            for (action, succ_state) in successors:
                val = max(val, self.alpha_beta_minimax(succ_state, depth - 1, alpha, beta))
                alpha = max(alpha, val)

                # if maximizer's best option >= minimizer's best option, prune
                # minimizer above will not choose maximizer's branch
                if alpha >= beta:
                    break

        elif isinstance(current_ent, Goblin):
            # Goblin Minimizer - starts comparing to infinity
            val = math.inf
            successors = sorted(successors, key=lambda pair: self.evaluation(pair[1]), reverse=False)

            for (action, succ_state) in successors:
                val = min(val, self.alpha_beta_minimax(succ_state, depth - 1, alpha, beta))
                beta = min(beta, val)

                # if minimizer's best option <= maximizer's best option, prune
                # maximizer above will not choose minimizer's branch anyway
                if beta <= alpha:
                    break

        return val

# WizardExpectimax: Evaluates using the average children value
class WizardExpectimax(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # if Wizard dead, return -100 (really bad score, should avoid)
        if len(state.get_all_entity_locations(Wizard)) == 0:
            return -100

        wiz = state.get_all_entity_locations(Wizard)[0]
        portal = state.get_all_tile_locations(Portal)[0]

        # if Wizard gets to portal, return 300 (really GOOD score, prefer!)
        if wiz == portal:
            return 300

        p_dist = abs(wiz.col - portal.col) + abs(wiz.row - portal.row)
        goblins = state.get_all_entity_locations(Goblin)

        # initially set closest_g_dist to a big number
        closest_g_dist = 100000
        for goblin in goblins:
            g_dist = abs(wiz.col - goblin.col) + abs(wiz.row - goblin.row)
            # if this goblin closer to the wizard than previous goblin, update
            if g_dist < closest_g_dist:
                closest_g_dist = g_dist

        # heuristic = manhattan(closest goblin - wizard) - manhattan(wizard - portal)
        return closest_g_dist - p_dist

    def is_terminal(self, state: GameState) -> bool:
        return (len(state.get_all_entity_locations(Wizard)) == 0 or
                (state.get_all_entity_locations(Wizard)[0] ==
                 state.get_all_tile_locations(Portal)[0]))

    def react(self, state: GameState) -> WizardMoves:
        values: dict[WizardMoves, float] = {}
        successors = self.get_successors(state)

        for (action, succ_state) in successors:
            values[action] = self.expectimax(succ_state, self.max_depth - 1)

        return max(values, key=values.get)

    def expectimax(self, state: GameState, depth: int):
        if (self.is_terminal(state) or depth == 0):
            return self.evaluation(state)
        successors = self.get_successors(state)
        current_ent = state.get_active_entity()
        ret = None

        if isinstance(current_ent, Wizard):
            # Wizard Maximizer - starts comparing to -infinity
            ret = -(math.inf)
            for (action, succ_state) in successors:
                ret = max(ret, self.expectimax(succ_state, depth - 1))
        elif isinstance(current_ent, Goblin):
            # Goblin averages the successor values
            successor_sum = 0.0
            for (action, succ_state) in successors:
                # add up all the successor children values
                successor_sum += self.expectimax(succ_state, depth - 1)

            # average the sum
            ret = successor_sum / len(successors)

        return ret
