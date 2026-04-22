from model import (
    Location,
    Portal,
    EmptyEntity,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import WizardSearchAgent
import heapq
from dataclasses import dataclass



class WizardDFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def next_search_expansion(self) -> GameState | None:
        # search_stack is empty, nowhere to expand
        if len(self.search_stack) == 0:
            return None

        current_state = self.search_stack.pop()
        if self.is_goal(current_state):
            # copy path of current_state, reverse since react() uses 'pop'
            reversed_path = self.paths[current_state].copy()
            reversed_path.reverse()
            self.plan = reversed_path
            return None
        else:
            # not goal, return the game state
            return self.search_to_game(current_state)

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # convert GameState's into SearchState's so can calculate
        src_state = self.game_to_search(source)
        tgt_state = self.game_to_search(target)

        if tgt_state in self.paths:
            # the target has already been visited
            return None
        else:
            # add tgt_state to the search_stack
            self.search_stack.append(tgt_state)

            # update paths[tgt_state] = path to source + new action
            self.paths[tgt_state] = self.paths[src_state].copy()
            self.paths[tgt_state].append(action)
            return


class WizardBFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def next_search_expansion(self) -> GameState | None:
        if len(self.search_stack) == 0:
            return None
        # oldest element gets removed, (first in first out)
        current_state = self.search_stack.pop(0)

        if self.is_goal(current_state):
            # copy and reverse, so react() will pop it in order
            reversed_path = self.paths[current_state].copy()
            reversed_path.reverse()
            self.plan = reversed_path
            return None
        else:
            # not goal, return the game state
            return self.search_to_game(current_state)

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        src_state = self.game_to_search(source)
        tgt_state = self.game_to_search(target)

        if tgt_state in self.paths:
            # target already visited
            return
        else:
            # first in, first out, insert new states at the end
            self.search_stack.append(tgt_state)

            # update paths[tgt_state] = path to source + new action
            self.paths[tgt_state] = self.paths[src_state].copy()
            self.paths[tgt_state].append(action)
            return



class WizardAstar(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, tuple[float, list[WizardMoves]]] = {}
    search_pq: list[tuple[float, SearchState]] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = 0, []
        self.search_pq = [(0, initial_search_state)]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def cost(self, source: GameState, target: GameState, action: WizardMoves) -> float:
        return 1

    def heuristic(self, target: GameState) -> float:
        cur_state = self.game_to_search(target)
        # return calculated manhanttan distance
        return (abs(cur_state.wizard_loc.col - cur_state.portal_loc.col) +
                abs(cur_state.wizard_loc.row - cur_state.portal_loc.row))

    def next_search_expansion(self) -> GameState | None:
        # return none if there is nothing in search_pq
        if len(self.search_pq) == 0:
            return None
        # unpack the least cost state
        priority, cur_state = heapq.heappop(self.search_pq)

        if self.is_goal(cur_state):
            # if is the goal, get path
            cur_cost, cur_path = self.paths[cur_state]
            reversed_path = cur_path.copy()
            reversed_path.reverse()
            # update plan
            self.plan = reversed_path
            return None
        else:
            # not goal, return the current state to find its possibilities
            return self.search_to_game(cur_state)

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # convert to SearchState
        src_state = self.game_to_search(source)
        tgt_state = self.game_to_search(target)

        # source's best known cost and path
        src_cost, src_path = self.paths[src_state]

        # new_cost = cost to source + cost of this move (1)
        new_cost = src_cost + self.cost(source, target, action)

        # build new path
        new_path = src_path.copy()
        new_path.append(action)

        # only keep this path if target unvisited OR new route is cheaper than old
        if tgt_state not in self.paths or new_cost < self.paths[tgt_state][0]:
            # update path[tgt_state] to be the new tuple
            self.paths[tgt_state] = (new_cost, new_path)

            # calculate new priority for the priorithy queue
            priority = new_cost + self.heuristic(target)
            # add new route to target to the priority queue
            heapq.heappush(self.search_pq, (priority, tgt_state))



class CrystalSearchWizard(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location
        crystals_left: frozenset[Location]

    paths: dict[SearchState, tuple[float, list[WizardMoves]]] = {}
    search_pq: list[tuple[float, SearchState]] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()
        crystal_locs = self.initial_game_state.get_all_entity_locations(Crystal)

        # Clear the entire board:
        # clear Wizard
        new_game_state = self.initial_game_state.replace_entity(initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity())
        # clear crystals
        for crystal in crystal_locs:
            new_game_state = new_game_state.replace_entity(crystal.row, crystal.col, EmptyEntity())
        # Refresh the board with the crystals left
        for crystal in search_state.crystals_left:
            new_game_state = new_game_state.replace_entity(crystal.row, crystal.col, Crystal())
        # new game state = new wizard
        new_game_state = (
            new_game_state.replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc))

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        crystals_left = frozenset(game_state.get_all_entity_locations(Crystal))
        return self.SearchState(wizard_loc, portal_loc, crystals_left)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = 0, []
        self.search_pq = [(0, initial_search_state)]

    # is_goal only true when no crystals remain and wizard is at portal
    def is_goal(self, state: SearchState) -> bool:
        return len(state.crystals_left) == 0 and state.wizard_loc == state.portal_loc

    # calculates the manhattan distance between loc1 to loc2
    def calculate_manhattan(self, loc1: Location, loc2: Location):

        return abs(loc1.col - loc2.col) + abs(loc1.row - loc2.row)

    # calculates the heuristic of crystals and portal to the wizard, uses MST
    def heuristic(self, target: GameState) -> float:
        cur_state = self.game_to_search(target)

        if len(cur_state.crystals_left) == 0:
            # return calculated manhanttan distance
            return self.calculate_manhattan(cur_state.wizard_loc, cur_state.portal_loc)

        # required points to go to = crystals + portal
        required_points = list(cur_state.crystals_left)
        required_points.append(cur_state.portal_loc)

        # wizard minimum edge to a required point
        wizard_min_connect = 100000
        for point in required_points:
            wizard_min_connect = min(wizard_min_connect, self.calculate_manhattan(cur_state.wizard_loc, point))

        # Minimum Spanning Tree
        tree_points = [required_points[0]] # points inside the tree
        out_points = list(required_points[1:]) # points outside the tree
        mst_cost = 0 # minimum spanning tree cost

        # while out_points is not empty:
        while(len(out_points) > 0):
            cheapest_edge_cost = 100000
            best_out_pt = None

            # calculate smallest Manhattan distance from tree points to points outside tree
            for t_pt in tree_points:
                for o_pt in out_points:
                    cost = self.calculate_manhattan(t_pt, o_pt)
                    if cost < cheapest_edge_cost:
                        best_out_pt = o_pt          # update closest out point
                        cheapest_edge_cost = cost

            # update MST cost
            mst_cost += cheapest_edge_cost
            if best_out_pt is not None:
                # update lists
                out_points.remove(best_out_pt)
                tree_points.append(best_out_pt)

        # heuristic = min distance wizard to a required point + cost of smallest distance of 2 crystals and/or portal
        return wizard_min_connect + mst_cost

    def cost(self, source: GameState, target: GameState, action: WizardMoves) -> float:
        return 1

    def next_search_expansion(self) -> GameState | None:
        if len(self.search_pq) == 0:
            # manhattan distance from wizard to portal
            return None
        priority, cur_state = heapq.heappop(self.search_pq)

        if(self.is_goal(cur_state)):
            # is goal, update paths, update plan
            cur_cost, cur_path = self.paths[cur_state]
            reversed_path = cur_path.copy()
            reversed_path.reverse()

            self.plan = reversed_path
            return None
        else:
            # otherwise, calculate this current state
            return self.search_to_game(cur_state)


    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        src_state = self.game_to_search(source)
        tgt_state = self.game_to_search(target)

        # source's best known cost and path
        src_cost, src_path = self.paths[src_state]

        # new_cost = cost to source + cost of this move (1)
        new_cost = src_cost + self.cost(source, target, action)

        # build new path
        new_path = src_path.copy()
        new_path.append(action)

        # only keep this path if target unvisited OR new route is cheaper than old
        if tgt_state not in self.paths or new_cost < self.paths[tgt_state][0]:
            # update path[tgt_state] to be the new tuple
            self.paths[tgt_state] = (new_cost, new_path)

            # calculate new priority for the priorithy queue
            priority = new_cost + self.heuristic(target)
            # add new route to target to the priority queue
            heapq.heappush(self.search_pq, (priority, tgt_state))


class SuboptimalCrystalSearchWizard(CrystalSearchWizard):

    # reduces node expansions, sacrificing guarantee of optimality
    def heuristic(self, target: GameState) -> float:
        return super().heuristic(target) * 1.5