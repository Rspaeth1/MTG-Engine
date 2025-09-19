import re
import uuid
import random
import math
import pandas as pd
import operator
from typing import List, Dict, Optional, Any, Set, Callable, Union
from enum import Enum, auto

# ==============================================================================
# --- ENUMS ---
# ==============================================================================

class Zone(Enum):
    HAND, COMMAND, GRAVEYARD, LIBRARY, EXILE, STACK, BATTLEFIELD, LIMBO = auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto()

class TurnPhase(Enum):
    BEGINNING, PRECOMBAT_MAIN, COMBAT, POSTCOMBAT_MAIN, ENDING = auto(), auto(), auto(), auto(), auto()

class TurnStep(Enum):
    UNTAP, UPKEEP, DRAW, MAIN, BEGINNING_OF_COMBAT, DECLARE_ATTACKERS, DECLARE_BLOCKERS, FIRST_STRIKE_DAMAGE, COMBAT_DAMAGE, END_OF_COMBAT, END, CLEANUP = auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto()

class Event(Enum):
    DEAL_DAMAGE, ZONE_CHANGE, DESTROY, SACRIFICE, DRAW_CARD, UNTAP, MILL_CARD = auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto()

class ReplacementType(Enum):
    QUANTITY_MODIFICATION, ZONE_REDIRECT, SKIP, ACTION_CHANGE = auto(), auto(), auto()

class CheckCondition(Enum):
    COLOR, TYPE = auto(), auto()

class SBAType(Enum):
    CHECK_PLAYER_LIFE, CHECK_LETHAL_DAMAGE, LEGEND_RULE, CHECK_COMMANDER_DAMAGE = auto(), auto(), auto(), auto()

class ActionType(Enum):
    PLAY_LAND, CAST_SPELL, ACTIVATE_ABILITY, PASS_PRIORITY, CAST_FROM_COMMAND_ZONE = auto(), auto(), auto(), auto(), auto()

class Speed(Enum):
    SORCERY, INSTANT = auto(), auto()

class RoundingType(Enum):
    UP, DOWN, NONE = auto(), auto(), auto()

class Comparator(Enum):
    LESS_THAN, GREATER_THAN, EQUAL, NOT_EQUAL = operator.lt, operator.gt, operator.eq, operator.ne

class Layer(Enum):
    COPY_EFFECTS = 1
    CONTROL_CHANGING = 2
    TEXT_CHANGING = 3
    TYPE_CHANGING = 4
    COLOR_CHANGING = 5
    ABILITY_CHANGING = 6
    POWER_TOUGHNESS_CDA = 7.1          # Characteristic-Defining Abilities
    POWER_TOUGHNESS_SETTING = 7.2      # Effects that set P/T
    POWER_TOUGHNESS_MODIFYING = 7.3    # Effects that modify P/T (e.g., +1/+1)
    POWER_TOUGHNESS_COUNTERS = 7.4     # Changes from counters
    POWER_TOUGHNESS_SWITCHING = 7.5    # Effects that switch P/T

class Duration(Enum):
    PERMANENT, UNTIL_END_OF_TURN = auto(), auto()

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def parse_mana_cost(cost_string: str) -> Dict[str, int]:
    if not cost_string: return {}
    cost = {'W': 0, 'U': 0, 'B': 0, 'R': 0, 'G': 0, 'C': 0, 'generic': 0}
    symbols = re.findall(r'\{([^}]+)\}', cost_string)
    for s in symbols:
        s = s.upper()
        if s.isdigit(): cost['generic'] += int(s)
        elif s in cost: cost[s] += 1
    return {k: v for k, v in cost.items() if v > 0}

def change_zone(card: 'Card', new_zone: Zone, index: int, new_controller: 'Player'):
    old_zone = card.zone
    controller = card.controller

    if old_zone == new_zone:
        return None

    if old_zone == Zone.COMMAND:
        controller.command_zone.remove(card)
    elif old_zone == Zone.EXILE:
        controller.exile.remove(card)
    elif old_zone == Zone.GRAVEYARD:
        controller.graveyard.remove(card)
    elif old_zone == Zone.HAND:
        controller.hand.remove(card)
    elif old_zone == Zone.LIBRARY:
        controller.library.remove(card)
    
    if new_zone == Zone.COMMAND:
        new_controller.command_zone.insert(index, card)
    elif new_zone == Zone.EXILE:
        new_controller.exile.insert(index, card)
    elif new_zone == Zone.GRAVEYARD:
        new_controller.graveyard.insert(index, card)
    elif new_zone == Zone.HAND:
        new_controller.hand.insert(index, card)
    elif new_zone == Zone.LIBRARY:
        new_controller.library.insert(index, card)

def condition_check(card: 'Card', condition: CheckCondition, checks: Set[Any]):
        match condition:
            case CheckCondition.COLOR:
                return checks.isdisjoint(set(card.colorIdentity))
            case CheckCondition.TYPE:
                return any(checks.isdisjoint(set(card.types)), checks.isdisjoint(set(card.subtypes)), checks.isdisjoint(set(card.supertypes)))

def quantity_multiplier(num: float, multiplier: float, rounding_type: RoundingType = RoundingType.NONE, minimum: int = None, maximum: int = None):
    num *= multiplier

    if minimum:
        num = max(num,minimum)
    if maximum:
        num = min(num,maximum)
    
    match rounding_type:
        case RoundingType.UP:
            return math.ceil(num)
        case RoundingType.DOWN:
            return math.floor(num)
        case RoundingType.NONE:
            return num
        
def quantity_addition(num: int, addition: int, minimum: int = None, maximum: int = None):
    num += addition

    if minimum:
        num = max(num,minimum)
    if maximum:
        num = min(num,maximum)
    
    return num


# ==============================================================================
# --- DECK LOADING ---
# ==============================================================================

class Deck:
    """A simple data object representing a valid Commander deck."""
    def __init__(self, commander: 'Card', library: List['Card']):
        self.commander = commander
        self.commander.zone = Zone.COMMAND
        self.library = library
    
    def __repr__(self) -> str:
        return f"Deck(Commander: {self.commander.name}, Library: {len(self.library)} cards)"

class DeckLoader:
    """Handles the entire process of reading, validating, and loading a deck, including DFCs."""
    def __init__(self, cards_path: str, legalities_path: str):
        self._cards_path = cards_path
        self._legalities_path = legalities_path # Stored for use in load_deck
        self._banned_uuids: Set[str] = set()

    def _clean_name(self, raw: str) -> str:
        # Precompiled patterns for trailing junk
        SETCODE_RE   = re.compile(r'\s*[A-Z]{2,6}[- ]?\d+[a-z]?\s*$', re.ASCII)  # e.g., "KTK-34", "MH1 164", "58p"
        PARENS_RE    = re.compile(r'\s*\([^)]*\)\s*$', re.ASCII)                 # e.g., "(PLST)", "(NEO)", "(6ED)"
        STAR_TAG_RE  = re.compile(r'\s*\*[^*]*\*\s*$', re.ASCII)                 # e.g., "*F*", "*Foil*"
        NUMBER_RE    = re.compile(r'\s*\d+[a-z]?\s*$', re.ASCII)                 # e.g., "15", or "58p" if no set code present

        name = raw.strip()
        # Iteratively strip trailing tokens until nothing more matches
        changed = True
        while changed:
            changed = False
            for pat in (STAR_TAG_RE, SETCODE_RE, NUMBER_RE, PARENS_RE):
                new = pat.sub('', name)
                if new != name:
                    name = new.strip()
                    changed = True
        return name

    def _read_decklist(self, filepath: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        commander_name = ""
        first_pass = True # To keep track of first line (commander)

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue

                # Quantity (handles "4", "4x", "4 x"); default to 1 if missing
                m = re.match(r'^\s*(\d+)\s*x?\s*(.+)$', s, flags=re.IGNORECASE)
                if m:
                    qty, rest = int(m.group(1)), m.group(2)
                else:
                    qty, rest = 1, s

                name = self._clean_name(rest)
                if not name:
                    continue

                # Grab commander name
                if first_pass:
                    commander_name = name
                    first_pass = False

                counts[name] = counts.get(name, 0) + qty

        
        return commander_name, counts

    def _fetch_card_data(self, decklist: Dict[str, int]):
        """Retrieves data for all faces of a specific list of card names."""
        names = decklist.keys()

        chunk_iter_a = pd.read_csv(self._cards_path, chunksize=5000, low_memory=True)
        all_cards_df = pd.concat(
            [chunk[chunk['name'].isin(names)] for chunk in chunk_iter_a]
        )

        # There may be a more efficient way to do this... but the cards db is precleaned except for some DFCs, this grabs only the first 2 occurrences
        all_cards_df = all_cards_df[
            (all_cards_df.groupby("name").cumcount() < 2)
            & ((all_cards_df["otherFaceIds"].notna())
            | (all_cards_df["otherFaceIds"] != ""))
        ]

        # Now duplicate cards that need it
        for name, count in decklist.items():
            if count > 1:
                row = all_cards_df.loc[(all_cards_df["name"] == name) 
                    & ((all_cards_df["otherFaceIds"].isna())
                    | (all_cards_df["otherFaceIds"] == "")) # Ignore DFCs for now (if adding other gamemodes where duplicates are allowed, would need a more sophisticated implementation)
                ].head(1).copy()

                if row.empty: continue # If nothing, continue

                dupes = pd.concat([row] * (count - 1), ignore_index=True) # Create duplicates
                all_cards_df = pd.concat([all_cards_df, dupes], ignore_index=True) # Concat

        # Check if we could find all cards
        for name in names:
            if not name in all_cards_df['name'].values:
                print(f"Error: Couldn't find card {name}")

        return all_cards_df

    def load_deck(self, filepath: str) -> Optional['Deck']:
        print(f"\n--- Loading deck: {filepath} ---")
        commander_name, decklist = self._read_decklist(filepath)

        if commander_name == "":
            return "Error getting commander name."

        # Fetch all required card data first
        all_faces_df = self._fetch_card_data(decklist)

        # Create a Card object for every face, keyed by UUID
        all_faces = [Card(row._asdict()) for row in all_faces_df.itertuples(index=False)]

        # Find commander
        commander = None
        for card in all_faces:
            if card.name == commander_name:
                commander = card
                break

        # Build the final list of cards
        deck = []
        seen = {}

        deck.append(commander) # Put commander first
        all_faces.remove(commander)

        for card in all_faces:
            if not card.is_dfc:
                deck.append(card)
            elif card.name not in seen: # again, this is pretty commander specific, other gamemodes would require more robust system
                deck.append(card)
                seen[card.name] = card
            else:
                seen_card = seen[card.name]
                old_spot = deck.index(seen_card)

                if seen_card.side == 'a':
                    deck[old_spot].otherFace = card
                else:
                    deck[old_spot] = card
                    card.otherFace = seen_card
        
        # Check for missing cards (e.g., typos in decklist)
        if len(deck) != 100:
            print(f"Error: Found {len(deck)} valid cards, but 100 are required.")
            return None

        # --- Validation Logic ---
        commander = deck[0]
        library = deck[1:]

        if not commander.can_be_commander:
            print(f"Validation Error: Card '{commander.name}' is an invalid commander.")
        
        commander_identity = set(commander.colorIdentity)
        seen = [commander]
        for card in library:
            card_identity = set(card.colorIdentity)
            if not card_identity.issubset(commander_identity):
                print(f"Validation Error: Card '{card.name}' is outside the commander's color identity.")
                return None
            
            # Commander cards uniqueness check
            if card in seen and not card.is_basic_land:
                print(f"Validation Error: Card '{card}' is in the deck more than once.")
            else:
                seen.append(card)

        
        print(f"Deck '{filepath}' loaded and validated successfully.")
        return Deck(commander=commander, library=library)
    
# ==============================================================================
# --- ACTIONS ---
# ==============================================================================

class Action:
    def __init__(self, player: 'Player', **kwargs):
        self.player = player
        self.data = kwargs
    def execute(self, game_state: 'GameState'):
        raise NotImplementedError

class PassPriorityAction(Action):
    def execute(self, game_state: 'GameState'): pass

class CastSpellAction(Action):
    def __init__(self, player: 'Player', card: 'Card'):
        super().__init__(player, card=card)
    def execute(self, game_state: 'GameState'):
        card = self.data['card']
        spell = Spell(card, cast_from_zone=Zone.HAND)
        game_state.stack.add(spell, self.player)
        self.player.hand.remove(card)

class CastFromCommandZoneAction(Action):
    def __init__(self, player: 'Player', card: 'Card'):
        super().__init__(player, card=card)
    def execute(self, game_state: 'GameState'):
        card = self.data['card']
        spell = Spell(card, cast_from_zone=Zone.COMMAND)
        game_state.stack.add(spell, self.player)
        self.player.commander_cast_count += 1
        
# ==============================================================================
# --- GAME SYSTEMS ---
# ==============================================================================

class manaPool:
    def __init__(self):
        self.pool = {'W': 0, 'U': 0, 'B': 0, 'R': 0, 'G': 0, 'C': 0}
    
    def add(self, color: str, amount: int):
        if color not in ['W', 'U', 'B', 'R', 'G', 'C']: return
        self.pool[color] += amount

    def remove(self, color: str, amount: int):
        if color not in ['W', 'U', 'B', 'R', 'G', 'C']: return
        self.pool[color] = min(0,self.pool[color]-amount)

class Player:
    def __init__(self, name: str, deck: Deck):
        self.name = name
        self.alive = True
        self.life = 40
        self.max_hand_size = 7
        self.lands_playable = 1 # Keep track of how many lands someone can play a turn
        self.hand: List['Card'] = []
        self.library: List['Card'] = []
        self.graveyard: List['Card'] = []
        self.exile: List['Card'] = []
        self.battlefield: List['Permanent'] = []
        self.limbo = List['Card'] = [] # Store cards here while they are spells on the stack or permanents on the battlefield
        self.command_zone: List['Card'] = []
        self.commander_cast_count = 0
        self.commander_damage: Dict[str, int] = {}
        self.deck = deck
        self.mana_pool = manaPool()

        self.commander = deck.commander
        self.library = deck.library
        self.index: int = -1
        self.mulligan_count: int = 0
        self.cards_drawn_this_turn = 0 # Remember to reset at turn end
        self.lands_played_this_turn = 0 # Remember to reset at turn end

        self.command_zone.append(self.commander)

    def shuffle_library(self):
        random.shuffle(self.library)
        print(f"{self.name} shuffles their library.")

    def draw_cards(self, num: int, first: bool):
        for _ in range(num):
            if self.library:
                card = self.library[-1]
                change_zone(card,Zone.HAND,-1,self) # Change zone
                self.cards_drawn_this_turn += 1 # Make sure to reset this after the turn
            else:
                # No need for SBA check here, if we go to draw and there are no cards, just kill player
                pass
        print(f"{self.name} draws {num} cards.")

    def discard_card(self, cards: List['Card']):
        for card in cards:
            change_zone(card, Zone.GRAVEYARD, -1, self) # Change zone, handle triggers/replacement there
        print(f"{self.name} discards {len(cards)} cards.")

    def mulligan(self, multiplayer: bool = True):
        print(f"{self.name} mulligans down to {new_hand_size}.")
        self.put_cards_on_top(self.hand)
        self.shuffle_library()
        self.mulligan_count += 1

        new_hand_size = self.max_hand_size - max(self.mulligan_count-1,0) if multiplayer else self.max_hand_size-self.mulligan_count # Give everyone a free mulligan if in multiplayer game
        self.draw_cards(new_hand_size)
        self._prompt_mulligan(new_hand_size)

    def put_cards_on_top(self, cards: List['Card']):
        for card in cards:
            change_zone(card, Zone.LIBRARY, -1, self)
        print(f"{self.name} puts {len(cards)} card{'s' if len(cards) != 1 else ''} on the top of their library.")
    
    def put_cards_on_bottom(self, cards: List['Card']):
        for card in cards:
            change_zone(card, Zone.LIBRARY, 0, self)
        print(f"{self.name} puts {len(cards)} card{'s' if len(cards) != 1 else ''} on the bottom of their library.")

    def get_decision(self, prompt: str, options: List[Any]) -> Any:
        print(f"  [DECISION] {self.name}: {prompt} {options}")
        if not options: return None
        return options[0] # Placeholder
    
    def can_pay_cost(self, cost: Dict[str, int]) -> bool:
        mana_pool = self.mana_pool.pool
        for color, amount in cost.items():
            if mana_pool[color] < amount:
                return False
        return True
    
    def _pay_cost(self, cost: Dict[str, int]):
        mana_pool = self.mana_pool.pool
        for color, amount in cost.items():
            if color != "generic":
                mana_pool.remove(color, amount)
            elif amount > 0:
                # GET INPUT FROM PLAYER ON HOW TO PAY FOR GENERIC COSTS    
                pass
    
    def _prompt_mulligan(self, new_hand_size: int = 7):
        decision = self.get_decision(f"Do you keep this hand?",  ['Keep', 'Mulligan'])
        if decision == 'Mulligan':
            self.mulligan()
        elif new_hand_size < self.max_hand_size:
            decision = self.get_decision(f"Which {self.max_hand_size-new_hand_size} card{'s' if self.max_hand_size-new_hand_size != 1 else ''} do you want to put on bottom?", [card.name for card in self.hand])
            self.put_cards_on_bottom(decision) # Needs revision

class Stack:
    def __init__(self):
        self._stack: List['Card'] = []

    def add(self, item: 'Card', player: 'Player'):
        item.owner = player
        item.controller = player
        print(f"  [STACK] {player.name} puts {item} on the stack.")
        self._stack.append(item)

    def remove(self, item: 'Card'):
        self._stack.remove(item)

    def resolve(self, game_state: 'GameState'):
        return self._stack.pop().resolve(game_state) if self._stack else None

    def is_empty(self) -> bool:
        return not self._stack

class ReplacementEffect:
    def __init__(self, event: Event, replacement_type: ReplacementType, replacement_conditions: tuple[Comparator, Dict], modification: Any):
        self.event = event
        self.replacementType = replacement_type
        self.replacementConditions = replacement_conditions
        self.modification = modification # Could be a callable, could be the new zone to go to, etc

class Modifier:
    def __init__(self, source, applies_to: Callable, effect: Callable, layer: Layer, duration: Duration = Duration.PERMANENT):
        self.source = source
        self.applies_to = applies_to
        self.effect = effect
        self.layer = layer
        self.duration = duration
        self.timestamp = self.game_state.get_next_timestamp() # Placeholder

 
# ==============================================================================
# --- CORE GAME OBJECTS ---
# ==============================================================================

class SBA:
    """A data object representing a single, targeted SBA check to be performed."""
    def __init__(self, sba_type: SBAType, **kwargs):
        self.type = sba_type
        self.data = kwargs

    def __eq__(self, other):
        if not isinstance(other, SBA):
            return False
        return self.type == other.type and self.data == other.data

    def __hash__(self):
        # A simple way to hash the dictionary for set/equality checks
        return hash((self.type, frozenset(self.data.items())))

    def __repr__(self):
        return f"SBA({self.type.name}, {self.data})"
    
class GameObject:
    def __init__(self, existing_uuid: Optional[str] = None):
        self.uuid: uuid.UUID = uuid.UUID(existing_uuid) if existing_uuid else uuid.uuid4()
        self.owner: Optional['Player'] = None
        self.controller: Optional['Player'] = None

    def __eq__(self, other):
        return isinstance(other, GameObject) and self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)

class Card(GameObject):
    def __init__(self, card_data: Dict[str, Any]):
        #super().__init__(card_data.get('uuid')) # If we want to use the pregenerated UUIDs
        super().__init__()
        
        def split_csv_string(s: Any) -> List[str]:
            if not isinstance(s, str) or pd.isna(s): return []
            return [item.strip() for item in str(s).split(',') if item.strip()]

        # --- Direct mapping from the database schema ---
        self.name: str = card_data.get('name', '')
        self.manaCost: str = card_data.get('manaCost', '')
        self.manaValue: float = float(card_data.get('manaValue', 0.0))
        self.colors: List[str] = split_csv_string(card_data.get('colors', ''))
        self.colorIdentity: List[str] = split_csv_string(card_data.get('colorIdentity', ''))
        self.supertypes: List[str] = split_csv_string(card_data.get('supertypes', ''))
        self.types: List[str] = split_csv_string(card_data.get('types', ''))
        self.subtypes: List[str] = split_csv_string(card_data.get('subtypes', ''))
        self.rarity: str = card_data.get('rarity', '')
        self.text: str = str(card_data.get('text', ''))
        self.flavorText: str = str(card_data.get('flavorText', ''))
        self.keywords: List[str] = split_csv_string(card_data.get('keywords', ''))
        self.power: str = str(card_data.get('power', ''))
        self.toughness: str = str(card_data.get('toughness', ''))
        self.loyalty: str = str(card_data.get('loyalty', ''))
        self.otherFaceIds: List[str] = split_csv_string(card_data.get('otherFaceIds', ''))
        self.zone = Zone.LIBRARY
        self.speed = Speed.INSTANT if 'Instant' in self.types or 'Flash' in self.keywords else Speed.SORCERY
        

        # --- Boolean Attributes ---
        self.can_be_commander = bool(card_data.get('commander',0))
        self.is_basic_land = True if 'Land' in self.types and 'Basic' in self.supertypes else False
        self.is_dfc = True if card_data.get('otherFaceIds','') != '' else False
        
        # --- Attributes for DFCs ---
        self.side: str = str(card_data.get('side','a'))
        self.activeSide = 'a'
        self.otherFace: Optional['Card'] = None

        # --- Parse Text ---
        effects = self._parse_text()
        self.activated_abilities = effects.get('ActivatedAbility',[])
        self.triggered_abilities = effects.get('TriggeredAbility',[])
        self.static_abilities = effects.get('StaticAbility',[])

    def flip_card(self): # NEEDS TO BE FIXED ----------------------------------------------------------
        if self.activeSide == 'a':
            self.activeSide = 'b'
            self.otherFace.activeSide = 'b'
        else:
            self.activeSide = 'a'
            self.otherFace.activeSide = 'a'

        temp_card = self.copy()
        self = self.otherFace
        self.otherFace = temp_card

    def look(self):
        print(f"Card name: {self.name}")
        print(f"Mana cost: {self.manaCost}")
        print(f"Color identity: {self.colorIdentity}")
        print(f"Supertypes: {' '.join(card_type for card_type in self.supertypes)}")
        print(f"Types: {' '.join(card_type for card_type in self.types)}")
        print(f"Subtypes: {' '.join(card_type for card_type in self.subtypes)}")
        print(f"Text: {self.text}")
        print(f"Flavor text: {self.flavorText}")
        if condition_check(self, CheckCondition.TYPE, set('Creature')):
            print(f"Power: {self.power}")
            print(f"Toughness: {self.toughness}")
        if condition_check(self, CheckCondition.TYPE, set('Planeswalker')): # Need to look into sieges / sagas, etc. On another note, figure out how to display counters
            print(f"Loyalty: {self.loyalty}")
        print(f"Dual faced card: {self.is_dfc}")

    def _parse_text(self):
        # CONCEPTUAL: This is where a sophisticated rules parser would go.
        # It would read the text and create ReplacementEffect objects.
        # Example for "If a creature would die, exile it instead."
        """ if "if a creature would die, exile it instead" in self.text.lower():
            return [
                ReplacementEffect(
                    original_event=TriggerType.DIES,
                    new_event_type=TriggerType.EXILED,
                    conditions=[lambda e: "Creature" in e.data.get('target', {}).get('types', [])],
                    description="If a creature would die, exile it instead"
                )
            ] """
        return {}

class Permanent(Card):
    def __init__(self, source_card: Card):
        super().__init__()
        self.source_card = source_card
        self.__dict__.update(vars(source_card)) # Gives this permanent all attributes of the source card

        self.damage_marked: int = 0
        self.tapped: bool = False
        self.untaps_in_untap: bool = True
        self.untappable: bool = True
        self.has_summoning_sickness: bool = "Creature" in self.source_card.types

class Ability(GameObject):
    def __init__(self, source, description: str, effect: Callable):
        super().__init__(source)
        self.description = description
        self.effect = effect

    def resolve(self, game_state: 'GameState'):
        print(f"  [STACK] Resolving ability: {self.description} from {self.source.name}")
        self.effect(self.source, game_state)

class StaticAbility(Ability):
    pass

class ActivatedAbility(Ability):
    pass

class TriggeredAbility(Ability):
    def __init__(self):
        super.__init__()

    def trigger(**params):
        pass
    

class Spell(Card):
    def __init__(self, source_card: Card, cast_from_zone: Zone, targets: List[Any] = []):
        super().__init__()
        self.cast_from_zone = cast_from_zone
        self.__dict__.update(vars(source_card)) # Gives this spell all attributes of the source card
        self.targets = targets
    
    @property
    def source_card(self) -> Card:
        return self.source

    def resolve(self, game_state: 'GameState'):
        print(f"  [STACK] Resolving spell: {self.name} (cast from {self.cast_from_zone.name})")
        is_permanent = any(t in self.source_card.types for t in ["Creature", "Artifact", "Enchantment", "Planeswalker", "Land"])
        if is_permanent:
            new_permanent = game_state.create_permanent_from_card(self.source_card, self.controller)
            game_state.event_manager.post_event(
                GameEvent(TriggerType.ENTERS_THE_BATTLEFIELD, permanent=new_permanent)
            )
        else:
            print(f"  [EFFECT] {self.source_card.name} has its effect.")
            self.owner.graveyard.append(self.source_card)
    
    @property
    def name(self):
        return self.source_card.name

# ==============================================================================
# --- PLAYER INPUT ---
# ==============================================================================

class InputHandler:
    @staticmethod
    def get_player_action(player: 'Player', game_state: 'GameState') -> Action:
        legal_action_types = game_state.get_legal_action_types(player)
        chosen_type = InputHandler.get_decision(player, "Choose an action type:", legal_action_types)
        
        if chosen_type == ActionType.CAST_SPELL:
            castable = [c for c in player.hand if "Land" not in c.types and player.can_pay_cost(c.mana_cost)] # Simplified
            card = InputHandler.get_decision(player, "Choose spell to cast:", castable)
            return CastSpellAction(player, card) if card else PassPriorityAction(player)
        
        # ... Other action types ...
        return PassPriorityAction(player)

    @staticmethod
    def get_decision(player: 'Player', prompt: str, options: List[Any]) -> Any:
        print(f"  [DECISION] {player.name}: {prompt} {options}")
        if not options: return None
        return options[0]

    @staticmethod
    def get_player_choice_for_trigger_order(player: 'Player', triggers: List[TriggeredAbility]) -> List[TriggeredAbility]:
        prompt = "Choose the order to stack your triggers (first in is last out):"
        options = list(triggers)
        print(f"  [DECISION] {player.name}: {prompt} {options}")
        return options
   
# ==============================================================================
# --- GAME ENGINE ---
# ==============================================================================

class GameState:
    def __init__(self, players: List[Player]):
        self.players = players
        self.num_players = len(players)
        self.active_player_index = 0
        self.current_phase: Optional[TurnPhase] = None
        self.current_step: Optional[TurnStep] = None
        self.stack = Stack()
        self.consecutive_passes = 0
        self.player_with_priority_index = 0
        self.sba_queue: List[SBA] = []
        self.event_manager = EventManager(self)
        self.pending_triggers: List[TriggeredAbility] = []
        self.active_modifiers: List[Modifier] = []
        self.current_turn = 0

    def setup_game(self):
        print("\n--- Setting up game ---")
        
        # 1. Randomly determine turn order
        print("Randomizing turn order...")
        self._set_turn_order()


        # 2. Players shuffle their libraries and draw opening hands
        self._loop_priority([ [Player.shuffle_library], [Player.draw_cards, {'num': 7}] ])
            

        # 3. Handle the mulligan process
        self._loop_priority([ [Player._prompt_mulligan, {}] ])

        # 4. Handle pre-game actions
        # placeholder for now
            
        print("\n--- Game setup complete ---\n")

    def game_loop(self):
        while True:
            self.current_turn += 1
            turn_num = self.current_turn

            print(f"\n--- Turn {turn_num} ---\n")

            self._loop_priority([ [self._take_turn, {}]  ])
            

            break # Placeholder ------------------------

    def _take_turn(self, player: Player):
        if not player.alive:
            return
        
        phases = [TurnPhase.BEGINNING, TurnPhase.PRECOMBAT_MAIN, TurnPhase.COMBAT, TurnPhase.POSTCOMBAT_MAIN, TurnPhase.ENDING] # Making this a list to make it more flexible for things such as adding more phases
        steps = {
            TurnPhase.BEGINNING: [TurnStep.UNTAP, TurnStep.UPKEEP, TurnStep.DRAW],
            TurnPhase.PRECOMBAT_MAIN: [TurnStep.MAIN],
            TurnPhase.COMBAT: [TurnStep.BEGINNING_OF_COMBAT, TurnStep.DECLARE_ATTACKERS, TurnStep.DECLARE_BLOCKERS, TurnStep.FIRST_STRIKE_DAMAGE, TurnStep.COMBAT_DAMAGE, TurnStep.END_OF_COMBAT],
            TurnPhase.POSTCOMBAT_MAIN: [TurnStep.MAIN], 
            TurnPhase.ENDING: [TurnStep.END, TurnStep.CLEANUP]
        }

        
        phase_num = 0
        while phase_num < len(phases):
            self.current_phase = phases[phase_num] # Set phase
            step_num = 0

            # 1. Advance step
            while step_num < len(steps[self.current_phase]):
                self.current_step = steps[self.current_phase][step_num] # Set step

                # 2. Turn-Based Actions
                self._turn_based_actions(player)

                # 3. Check for begginning of step triggers
                # Post begginning of step trigger

                # 4. Players receive priority
                #If we are in untap, first strike, combat damage, or cleanup, no priority is received
                if self.current_step not in [TurnStep.UNTAP, TurnStep.FIRST_STRIKE_DAMAGE, TurnStep.COMBAT_DAMAGE, TurnStep.CLEANUP]:
                    #self._loop_priority([[]]) --- get decision based on legal actions
                    pass

                # 5. Check for end of step triggers
                # Post end of step trigger

                # Go to next step
                step_num += 1
            
            # Go to next phase
            phase_num += 1
            
    def _turn_based_actions(self, player):
        step = self.current_step

        match step: # These are the only steps where TBAs take place
            case TurnStep.UNTAP:
                # modify this to post an event to possibly replace the untapping with something (i.e., winter orb)
                self.untap(player.battlefield)
            case TurnStep.DRAW:
                player.draw_cards(1)
            case TurnStep.DECLARE_ATTACKERS:
                # Active player declares attackers
                pass
            case TurnStep.DECLARE_BLOCKERS:
                # Players being attacked declare blockers
                pass
            case TurnStep.FIRST_STRIKE_DAMAGE:
                # Deal first strike damage
                pass
            case TurnStep.COMBAT_DAMAGE:
                # Deal combat damage
                pass
            case TurnStep.CLEANUP:
                if len(player.hand) > player.max_hand_size:
                    # Prompt player to choose cards to discard
                    pass
                # Remove all damage marked on permanents
                # End until end of turn effects

    def untap(self, permanents: List['Permanent']):
        for permanent in permanents:
            # Check if we can untap
            if (self.current_step == TurnStep.UNTAP and permanent.untaps_in_untap and permanent.untappable) or (self.current_step != TurnStep.UNTAP and permanent.untappable):
                # Modify this to post triggers on untaps
                permanent.untap()
    
    def _insert_step(self, steps: List[TurnStep], StepType: TurnStep, index: int):
        steps.insert(index + 1, StepType)
    
    def _insert_phase(self, phases: List[TurnPhase], PhaseType: TurnPhase, index: int):
        phases.insert(index + 1, PhaseType)


    def _set_turn_order(self):
        random.shuffle(self.players)
        print(f"Turn order: {' - '.join(player.name for player in self.players)}")
    
    def _loop_priority(self, actions: List[List[Any]]):
        active_player = self.active_player_index
        num_players = self.num_players

        for i in range(active_player, active_player + num_players):
            current_player = self.players[i % num_players] # Loop through each player starting with active player
            for item in actions: # Loop through each method / action we want them to take
                action = item[0]
                if len(item) > 1:
                    params = item[1]
                else:
                    params = {}

                action(current_player, **params) # Execute
                        
    def create_permanent_from_card(self, card: Card, controller: 'Player') -> Permanent:
        new_permanent = Permanent(card, self)
        new_permanent.owner = card.owner
        new_permanent.controller = controller
        controller.battlefield.append(new_permanent)
        print(f"    -> {new_permanent.name} enters the battlefield under {controller.name}'s control.")
        return new_permanent
    
    def destroy_permanent(self, permanent: Permanent):
        permanent.controller.battlefield.remove(permanent)

    def get_characteristics(self, permanent: Permanent) -> Characteristics:
        chars = Characteristics(permanent)
        relevant_modifiers = [mod for mod in self.active_modifiers if mod.applies_to(permanent)]
        relevant_modifiers.sort(key=lambda m: m.layer.value)
        for mod in relevant_modifiers:
            mod.effect(chars)
        return chars

    def has_flash_permission(self, player: 'Player', card: Card) -> bool:
        return False # Hook for modifier system

    def get_legal_action_types(self, player: 'Player') -> List[ActionType]:
        # Placeholder for full legal action checking
        action_types = [ActionType.PASS_PRIORITY]
        # A full implementation would check timing, permissions, etc.
        if any("Land" not in c.types for c in player.hand):
            action_types.insert(0, ActionType.CAST_SPELL)
        return action_types # Note that action_types doesn't return every possible action, but just the possible actions that the player can take

    def validate_action(self, action: Action) -> bool:
        player = action.player
        if isinstance(action, CastSpellAction):
            card = action.data['card']
            if card not in player.hand: return False
            # The can_pay_cost check here is using the placeholder method.
            if not player.can_pay_cost({}): return False
            # A full implementation would also check for timing permission (sorcery vs. instant speed).
        return True

    def _check_state_based_actions(self):
        """
        Continuously processes the SBA queue until the game state is stable.
        This is the core of the SBA engine.
        """
        while True:
            # Move the current queue to a processing list and clear the main queue.
            # This handles cases where resolving an SBA queues up another one.
            if not self.sba_queue:
                break # No checks are pending, so the state is stable.

            checks_to_process = list(self.sba_queue)
            self.sba_queue.clear()
            
            actions_taken = False
            for check in checks_to_process:
                if check.type == SBAType.CHECK_PLAYER_LIFE:
                    player = check.data['player']
                    if player.life <= 0:
                        print(f"  [SBA] {player.name} has {player.life} life and loses the game.")
                        # Future implementation: move player to a 'lost' state.
                        actions_taken = True

                elif check.type == SBAType.CHECK_LETHAL_DAMAGE:
                    permanent = check.data['permanent']
                    # Ensure the permanent is still on the battlefield before checking.
                    if permanent in permanent.controller.battlefield and "Creature" in permanent.types:
                        if permanent.damage_marked >= permanent.toughness:
                            print(f"  [SBA] {permanent.name} has lethal damage ({permanent.damage_marked}/{permanent.toughness}).")
                            self.destroy_permanent(permanent)
                            actions_taken = True
                
                elif check.type == SBAType.CHECK_COMMANDER_DAMAGE:
                    player = check.data['player']
                    for commander_uuid, damage in player.commander_damage.items():
                        if damage >= 21:
                            print(f"  [SBA] {player.name} has taken {damage} commander damage and loses the game.")
                            # Future implementation: move player to a 'lost' state.
                            actions_taken = True
                            break # Player has already lost, no need to check other commanders.
                
                elif check.type == SBAType.CHECK_LEGEND_RULE:
                    permanent = check.data['permanent']
                    for perm in permanent.controller.battlefield:
                        if perm.name == permanent.name:
                            keep_choice = InputHandler.get_decision(permanent.controller, "Choose which creature to keep", [perm, permanent])
                            if keep_choice == permanent:
                                permanent.d
            
            # If a full pass resulted in no actions, the state is stable.
            if not actions_taken:
                break

    def _stack_pending_triggers(self):
        if not self.pending_triggers: return
        ap_index = self.active_player_index
        for i in range(len(self.players)):
            player_index = (ap_index + i) % len(self.players)
            player = self.players[player_index]
            player_triggers = [t for t in self.pending_triggers if t.controller == player]
            if not player_triggers: continue
            
            ordered_triggers = player_triggers
            if len(player_triggers) > 1:
                ordered_triggers = InputHandler.get_player_choice_for_trigger_order(player, player_triggers)
            
            for trigger in ordered_triggers:
                self.stack.add(trigger, player)
                self.pending_triggers.remove(trigger)
        
        self.pending_triggers.clear()

    def run_priority_loop(self):
        while True:
            self._check_state_based_actions()
            self._stack_pending_triggers()

            if self.consecutive_passes >= len(self.players):
                if self.stack.is_empty():
                    break
                else:
                    self.stack.resolve(self)
                    self.consecutive_passes = 0
                    self.player_with_priority_index = self.active_player_index
                    continue

            current_player = self.players[self.player_with_priority_index]
            chosen_action = InputHandler.get_player_action(current_player, self)
            
            if isinstance(chosen_action, PassPriorityAction):
                self.consecutive_passes += 1
                self.player_with_priority_index = (self.player_with_priority_index + 1) % len(self.players)
            else:
                self.consecutive_passes = 0
                chosen_action.execute(self)
                self.player_with_priority_index = self.active_player_index
                
    def _execute_step_actions(self):
        # Placeholder for automatic step/phase actions
        if self.current_step == TurnStep.CLEANUP:
            expired_mods = [m for m in self.active_modifiers if m.duration == Duration.UNTIL_END_OF_TURN]
            for mod in expired_mods:
                self.active_modifiers.remove(mod)



