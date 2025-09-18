import re
import uuid
import random
import pandas as pd
from typing import List, Dict, Optional, Any, Set, Callable, Union, Iterable
from enum import Enum, auto

# ==============================================================================
# --- ENUMS ---
# ==============================================================================

class Zone(Enum):
    HAND, COMMAND, GRAVEYARD, LIBRARY, EXILE, STACK, BATTLEFIELD = auto(), auto(), auto(), auto(), auto(), auto(), auto()

class TurnPhase(Enum):
    BEGINNING, PRECOMBAT_MAIN, COMBAT, POSTCOMBAT_MAIN, ENDING = auto(), auto(), auto(), auto(), auto()

class TurnStep(Enum):
    UNTAP, UPKEEP, DRAW, MAIN, BEGINNING_OF_COMBAT, DECLARE_ATTACKERS, DECLARE_BLOCKERS, FIRST_STRIKE_DAMAGE, COMBAT_DAMAGE, END_OF_COMBAT, END, CLEANUP = auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto()

class TriggerType(Enum):
    DAMAGE_DEALT, ENTERS_THE_BATTLEFIELD, DIES, LEAVES_THE_BATTLEFIELD, DESTROYED, SACRIFICED = auto(), auto(), auto(), auto(), auto(), auto()

class SBAType(Enum):
    CHECK_PLAYER_LIFE, CHECK_LETHAL_DAMAGE, LEGEND_RULE, CHECK_COMMANDER_DAMAGE = auto(), auto(), auto(), auto()

class ActionType(Enum):
    PLAY_LAND, CAST_SPELL, ACTIVATE_ABILITY, PASS_PRIORITY, CAST_FROM_COMMAND_ZONE = auto(), auto(), auto(), auto(), auto()

class Speed(Enum):
    SORCERY, INSTANT = auto(), auto()

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

def change_zone(card: 'Card', new_zone: Zone, index: int, new_controller: 'Player') -> Zone:
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
        # No pre-indexing of the main card file happens here.

    def _read_decklist(self, filepath: str) -> List[str]:
        with open(filepath, 'r') as f: lines = [line.strip() for line in f if line.strip()]
        return [re.sub(r'^\d+\s*x?\s*', '', line, flags=re.IGNORECASE) for line in lines]

    def _fetch_card_data(self, names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Efficiently retrieves data for all faces of a specific list of card names."""
        # This method is good as-is from your example. It correctly finds front
        # faces by name and back faces by UUID in two passes.
        chunk_iter_a = pd.read_csv(self._cards_path, chunksize=5000, low_memory=True)
        front_faces_df = pd.concat(
            [chunk[chunk['name'].isin(names)] for chunk in chunk_iter_a]
        )
        other_face_uuids = set()
        if 'otherFaceIds' in front_faces_df.columns:
            for face_ids in front_faces_df['otherFaceIds'].dropna():
                for uuid_str in str(face_ids).split(','):
                    if uuid_str: other_face_uuids.add(uuid_str)
        all_card_data_df = front_faces_df
        if other_face_uuids:
            chunk_iter_b = pd.read_csv(self._cards_path, chunksize=5000, low_memory=True)
            back_faces_df = pd.concat(
                [chunk[chunk['uuid'].isin(other_face_uuids)] for chunk in chunk_iter_b]
            )
            all_card_data_df = pd.concat([front_faces_df, back_faces_df]).drop_duplicates(subset=['uuid'])
        return {row['uuid']: row for row in all_card_data_df.to_dict('records')}

    def load_deck(self, filepath: str) -> Optional['Deck']:
        print(f"\n--- Loading deck: {filepath} ---")
        card_names_from_file = self._read_decklist(filepath)
        if len(card_names_from_file) != 100:
            print("Error: Deck must have 100 cards.")
            return None

        # Fetch all required card data first
        all_faces_data_map = self._fetch_card_data(list(set(card_names_from_file)))
        if not all_faces_data_map: return None

        # Create a Card object for every face, keyed by UUID
        card_objects_by_uuid = {uuid: Card(data) for uuid, data in all_faces_data_map.items()}

        # Link the double-faced cards together
        for card_obj in card_objects_by_uuid.values():
            if card_obj.otherFaceIds:
                other_face_uuid = card_obj.otherFaceIds[0]
                if other_face_uuid in card_objects_by_uuid:
                    card_obj.otherFace = card_objects_by_uuid[other_face_uuid]
                    card_obj.side = card_obj.side or 'a'
                    card_obj.otherFace.side = 'b'

        # --- REFINED DECK ASSEMBLY LOGIC ---
        # Create a temporary name-to-uuid map from the cards we just fetched.
        # This solves the bug where _name_to_uuid didn't exist.
        name_to_uuid_map = {card.name.lower(): card.uuid for card in card_objects_by_uuid.values()}

        # Build the final list of cards in the order they appeared in the file.
        final_deck_cards = []
        for name in card_names_from_file:
            uuid = name_to_uuid_map.get(name.lower())
            if uuid and uuid in card_objects_by_uuid:
                final_deck_cards.append(card_objects_by_uuid[uuid])
        
        # Check for missing cards (e.g., typos in decklist)
        if len(final_deck_cards) != 100:
            print(f"Error: Found {len(final_deck_cards)} valid cards, but 100 are required.")
            return None

        # --- Validation Logic ---
        commander = final_deck_cards[0]
        library = final_deck_cards[1:]

        if "Legendary" not in commander.supertypes or "Creature" not in commander.types:
            print(f"Validation Error: Commander '{commander.name}' is not a Legendary Creature.")
            return None
        
        commander_identity = set(commander.colorIdentity)
        for card in library:
            card_identity = set(card.colorIdentity)
            if not card_identity.issubset(commander_identity):
                print(f"Validation Error: Card '{card.name}' is outside the commander's color identity.")
                return None
        
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

class Player:
    def __init__(self, name: str, deck: Deck):
        self.name = name
        self.alive = True
        self.life = 40
        self.max_hand_size = 7
        self.hand: List['Card'] = []
        self.library: List['Card'] = []
        self.graveyard: List['Card'] = []
        self.exile: List['Card'] = []
        self.battlefield: List['Permanent'] = []
        self.command_zone: List['Card'] = []
        self.commander_cast_count = 0
        self.commander_damage: Dict[str, int] = {}
        self.land_played_this_turn = False
        self.deck = deck

        self.commander = deck.commander
        self.library = deck.library
        self.index: int = -1
        self.mulligan_count: int = 0
        self.cards_drawn_this_turn = 0

        self.command_zone.append(self.commander)

    def shuffle_library(self):
        random.shuffle(self.library)
        print(f"{self.name} shuffles their library.")

    def draw_cards(self, num: int):
        for _ in range(num):
            if self.library:
                card = self.library[-1]
                change_zone(card,Zone.HAND,-1,self) # Change zone, handle triggers/replacement there
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
        return True # Placeholder
    
    def _prompt_mulligan(self, new_hand_size: int = 7):
        decision = self.get_decision(f"Do you keep this hand?",  ['Keep', 'Mulligan'])
        if decision == 'Mulligan':
            self.mulligan()
        elif new_hand_size < self.max_hand_size:
            decision = self.get_decision(f"Which {self.max_hand_size-new_hand_size} card{'s' if self.max_hand_size-new_hand_size != 1 else ''} do you want to put on bottom?", [card.name for card in self.hand])
            self.put_cards_on_bottom(decision) # Needs revision
    

class Stack:
    def __init__(self):
        self._stack: List[Stackable] = []

    def add(self, item: 'Stackable', player: 'Player'):
        item.owner = player
        item.controller = player
        print(f"  [STACK] {player.name} puts {item} on the stack.")
        self._stack.append(item)

    def remove(self, item: 'Stackable') -> 'Stackable':
        self._stack.remove(item)
        return item

    def resolve(self, game_state: 'GameState'):
        return self._stack.pop().resolve(game_state) if self._stack else None

    def is_empty(self) -> bool:
        return not self._stack

class GameEvent:
    def __init__(self, event_type: TriggerType, **kwargs):
        self.type = event_type
        self.data = kwargs
        self.is_replaced = False

class ReplacementEffect:
    def __init__(self, original_event: TriggerType, new_event_type: TriggerType, conditions: List[Callable], description: str):
        self.original_event_type = original_event
        self.new_event_type = new_event_type
        # Conditions are functions that check the event data, e.g., lambda event: "Creature" in event.data['target'].types
        self.conditions = conditions
        self.description = description

    def check_conditions(self, event: GameEvent) -> bool:
        """Checks if this replacement effect applies to the given event."""
        if event.type != self.original_event_type:
            return False
        return all(cond(event) for cond in self.conditions)

    def create_new_event(self, original_event: GameEvent) -> GameEvent:
        """Creates the new event based on the original."""
        # This logic would be more complex, transferring relevant data from old to new event
        new_data = original_event.data.copy()
        print(f"  [REPLACE] '{self.description}' is replacing {original_event.type.name} with {self.new_event_type.name}.")
        return GameEvent(self.new_event_type, **new_data)

class EventManager:
    def __init__(self, game_state: 'GameState'):
        self.game_state = game_state
        self.registered_triggers: Dict[TriggerType, List[Callable]] = {}

    def _get_applicable_replacements(self, event: GameEvent) -> List[ReplacementEffect]:
        """Finds all active replacement effects that could apply to an event."""
        effects = []
        for modifier in self.game_state.active_modifiers:
            if modifier.replacement_effect and modifier.replacement_effect.check_conditions(event):
                effects.append(modifier.replacement_effect)
        return effects

    def post_event(self, event: GameEvent):
        """
        The main event pipeline, now with true event replacement.
        """
        # 1. Check for and apply any replacement effects
        applicable_replacements = self._get_applicable_replacements(event)

        if applicable_replacements:
            # If multiple replacements apply, the affected player chooses one.
            # This requires a call to the InputHandler.
            chosen_replacement = applicable_replacements[0] # Placeholder
            
            # The original event is replaced by a new one.
            event = chosen_replacement.create_new_event(event)

        # 2. The final event occurs. Now, check for triggered abilities for the NEW event.
        if event.type in self.registered_triggers:
            for trigger_factory in self.registered_triggers[event.type]:
                triggered_ability = trigger_factory(event)
                if triggered_ability:
                    self.game_state.pending_triggers.append(triggered_ability)
                    print(f"  [EVENT] {event.type.name} caused '{triggered_ability.description}' to trigger.")

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
    """Represents the printed, immutable data of a Magic card, now with DFC support."""
    def __init__(self, card_data: Dict[str, Any]):
        super().__init__(card_data.get('uuid'))
        
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
        
        # --- Attributes for DFCs ---
        self.side: Optional[str] = card_data.get('side')
        self.activeSide = 'a'
        self.otherFace: Optional['Card'] = None

    def _parse_replacement_effects(self, text: str) -> List[ReplacementEffect]:
        # CONCEPTUAL: This is where a sophisticated rules parser would go.
        # It would read the text and create ReplacementEffect objects.
        # Example for "If a creature would die, exile it instead."
        if "if a creature would die, exile it instead" in text.lower():
            return [
                ReplacementEffect(
                    original_event=TriggerType.DIES,
                    new_event_type=TriggerType.EXILED,
                    conditions=[lambda e: "Creature" in e.data.get('target', {}).get('types', [])],
                    description="If a creature would die, exile it instead"
                )
            ]
        return []

    def __repr__(self) -> str:
        if self.side:
            return f"{self.name} (Side {self.side.upper()})"
        return f"{self.name}"

class Characteristics:
    def __init__(self, permanent: 'Permanent'):
        self.power = int(permanent.source_card.printed_power or 0) if permanent.source_card.printed_power.isdigit() else 0
        self.toughness = int(permanent.source_card.printed_toughness or 0) if permanent.source_card.printed_toughness.isdigit() else 0
        self.keywords = set(permanent.source_card.keywords)
        self.types = set(permanent.source_card.types)

class Permanent(GameObject):
    def __init__(self, source_card: Card, game_state: 'GameState'):
        super().__init__()
        self.source_card = source_card
        self._game_state = game_state
        self.damage_marked: int = 0
        self.tapped: bool = False
        self.untaps_in_untap: bool = True
        self.untappable: bool = True
        self.has_summoning_sickness: bool = "Creature" in self.source_card.types

    @property
    def name(self) -> str:
        return self.source_card.name

    @property
    def power(self) -> int:
        return self._game_state.get_characteristics(self).power
    
    @property
    def toughness(self) -> int:
        return self._game_state.get_characteristics(self).toughness

    @property
    def keywords(self) -> Set[str]:
        return self._game_state.get_characteristics(self).keywords

    @property
    def types(self) -> Set[str]:
        return self._game_state.get_characteristics(self).types

    def __repr__(self):
        base = f"{self.name}"
        if "Creature" in self.types:
            base += f" ({self.power}/{self.toughness})"
        if self.tapped: base += " (Tapped)"
        return base

class Stackable(GameObject):
    def __init__(self, source):
        super().__init__()
        self.source = source
    def resolve(self, game_state: 'GameState'):
        raise NotImplementedError
    def __repr__(self):
        return f"{self.__class__.__name__}(source={self.source.name})"

class Ability(Stackable):
    pass

class TriggeredAbility(Ability):
    def __init__(self, source, description: str, effect: Callable):
        super().__init__(source)
        self.description = description
        self.effect = effect
    def resolve(self, game_state: 'GameState'):
        print(f"  [STACK] Resolving trigger: {self.description} from {self.source.name}")
        self.effect(self.source, game_state)

class Spell(Stackable):
    def __init__(self, source_card: Card, cast_from_zone: Zone):
        super().__init__(source_card)
        self.cast_from_zone = cast_from_zone
    
    @property
    def source_card(self) -> Card:
        return self.source

    def resolve(self, game_state: 'GameState'):
        print(f"  [STACK] Resolving spell: {self.source_card.name} (cast from {self.cast_from_zone.name})")
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

    def setup_game(self):
        print("\n--- Setting up game ---")
        
        # 1. Randomly determine turn order
        random.shuffle(self.players)
        print("Randomizing turn order...")
        for i, player in enumerate(self.players):
            player.index = i
            print(f"  Turn {i+1}: {player.name}")

        # 2. Players shuffle their libraries and draw opening hands
        for player in self.players:
            player.shuffle_library()
            player.draw_cards(7)
            print(f"{player.name} shuffles their library and draws their opening hand.")

        # 3. Handle the mulligan process
        self._handle_mulligans()

        # 4. Handle pre-game actions
        # placeholder for now
            
        print("--- Game setup complete ---\n")

    def _handle_mulligans(self):
        """Manages the London mulligan process for all players."""
        # Phase 1: Each player decides whether to keep or mulligan.
        for player in self.players:
            while True:
                prompt = f"{player.name}, do you keep this hand of {len(player.hand)}?"
                # In a real game, this would show the hand.
                choice = InputHandler.get_decision(player, prompt, ["Keep", "Mulligan"])
                
                if choice == "Keep":
                    break
                
                # Player chose to mulligan
                player.mulligan_count += 1
                print(f"  [MULLIGAN] {player.name} mulligans to {7 - player.mulligan_count + 1}.")
                player.return_hand_to_library()
                player.draw_cards(7)
                
                if len(player.hand) == 0: # Cannot mulligan a hand of 0
                    break

        # Phase 2: Players who mulliganed now put cards on the bottom of their library.
        print("--- Resolving Mulligans ---")
        for player in self.players:
            if player.mulligan_count - 1 > 0:
                num_to_bottom = player.mulligan_count - 1
                prompt = f"{player.name}, choose {num_to_bottom} card(s) to put on the bottom of your library."
                
                for i in range(num_to_bottom):
                    card_to_bottom = InputHandler.get_decision(player, f"{prompt} ({num_to_bottom - i} remaining)", player.hand)
                    if card_to_bottom:
                        player.hand.remove(card_to_bottom)
                        player.library.append(card_to_bottom) # Add to the bottom
                        print(f"  [MULLIGAN] {player.name} puts a card on the bottom. Hand size: {len(player.hand)}")

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

