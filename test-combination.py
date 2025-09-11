import os
import json
import math
import random
import re
from collections import Counter, defaultdict
import csv
from typing import Dict, List, Optional, Sequence, Tuple

import torch as t
from nnsight import LanguageModel
from transformers import AutoTokenizer

from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
from config import lm

GPU = "0"
LAYER = 8
E = 16
# Optional explicit paths; if None, we'll auto-discover from dictionaries/
SCALE_MODEL_PATH: Optional[str] = f"dictionaries/MultiExpert_Scale_64_{E}/{LAYER}.pt"
PLAIN_MODEL_PATH: Optional[str] = f"dictionaries/MultiExpert_64_{E}/{LAYER}.pt"

OUTPUT_ROOT = f"feature_combo_overlap_both_{E}"

TOP_N_FEATURES = 8
NUM_PAST_VERBS = 1500
SEED = 0

# Optional: override token list from a file (one token per line). If set, NUM_PAST_VERBS is ignored.
TOKENS_FILE: Optional[str] = None

PLOT_OVERALL = True
PLOT_PER_EXPERT = False
PLOT_FEATURE_SCATTER = True

MIN_FEATURE_COUNT = 5  # only features seen in >= this many tokens get summarized
TOP_FEATURES_TO_LIST = 50  # for optional top tables

# How many most-different tokens to export (by lowest Scale vs Plain Jaccard)
DIFF_TOP_K = 200
JACCARD_THRESHOLD = 0.25  # also export all tokens at or below this threshold


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_past_verbs(max_count: int = 1500) -> List[str]:
    verbs: List[str] = []
    try:
        import importlib.util

        ta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test-activate.py'))
        if os.path.exists(ta_path):
            spec = importlib.util.spec_from_file_location("test_activate_mod", ta_path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, 'TOKEN_CATEGORIES') and 'verbs_past' in mod.TOKEN_CATEGORIES:
                verbs = list(mod.TOKEN_CATEGORIES['verbs_past'])
    except Exception:
        pass

    if not verbs:
        # Fallback minimal list; user should supply a file or rely on test-activate list for full 1500
        verbs = [
        'walked', 'talked', 'played', 'worked', 'studied', 'learned', 'taught', 'helped', 'called', 'asked',
        'answered', 'listened', 'watched', 'looked', 'saw', 'heard', 'felt', 'touched', 'smelled', 'tasted',
        'ate', 'drank', 'cooked', 'cleaned', 'washed', 'dried', 'folded', 'packed', 'unpacked', 'moved',
        'carried', 'lifted', 'pushed', 'pulled', 'threw', 'caught', 'kicked', 'hit', 'punched', 'slapped',
        'hugged', 'kissed', 'loved', 'liked', 'hated', 'feared', 'worried', 'hoped', 'wished', 'dreamed',
        'slept', 'woke', 'stood', 'sat', 'lay', 'fell', 'jumped', 'ran', 'jogged', 'skipped', 'crawled',
        'climbed', 'swam', 'flew', 'drove', 'rode', 'sailed', 'traveled', 'visited', 'stayed', 'left',
        'arrived', 'departed', 'returned', 'came', 'went', 'entered', 'exited', 'opened', 'closed', 'locked',
        'unlocked', 'started', 'stopped', 'finished', 'began', 'continued', 'paused', 'waited', 'hurried', 'rushed',
        'delayed', 'postponed', 'canceled', 'planned', 'organized', 'arranged', 'scheduled', 'booked', 'reserved',
        
        'thought', 'remembered', 'forgot', 'understood', 'confused', 'realized', 'recognized', 'discovered', 'explored', 'investigated',
        'analyzed', 'compared', 'contrasted', 'evaluated', 'assessed', 'judged', 'decided', 'chose', 'selected', 'preferred',
        'considered', 'pondered', 'reflected', 'meditated', 'concentrated', 'focused', 'imagined', 'visualized', 'predicted', 'anticipated',
        'expected', 'assumed', 'supposed', 'guessed', 'estimated', 'calculated', 'measured', 'counted', 'numbered', 'added',
        'subtracted', 'multiplied', 'divided', 'solved', 'figured', 'reasoned', 'concluded', 'deduced', 'inferred', 'implied',
        
        'spoke', 'whispered', 'shouted', 'screamed', 'yelled', 'cried', 'laughed', 'giggled', 'chuckled', 'smiled',
        'frowned', 'grimaced', 'winked', 'nodded', 'shook', 'gestured', 'pointed', 'waved', 'signaled', 'indicated',
        'expressed', 'communicated', 'conveyed', 'transmitted', 'delivered', 'announced', 'declared', 'proclaimed', 'stated', 'mentioned',
        'noted', 'remarked', 'commented', 'observed', 'noticed', 'spotted', 'detected', 'identified', 'located', 'found',
        'searched', 'seeked', 'hunted', 'pursued', 'chased', 'followed', 'tracked', 'traced', 'monitored', 'supervised',
        
        'created', 'made', 'built', 'constructed', 'assembled', 'manufactured', 'produced', 'generated', 'formed', 'shaped',
        'molded', 'carved', 'sculpted', 'painted', 'drew', 'sketched', 'designed', 'drafted', 'planned', 'blueprinted',
        'invented', 'developed', 'innovated', 'improved', 'enhanced', 'modified', 'altered', 'changed', 'transformed', 'converted',
        'adapted', 'adjusted', 'refined', 'perfected', 'polished', 'finished', 'completed', 'accomplished', 'achieved', 'succeeded',
        'failed', 'attempted', 'tried', 'tested', 'experimented', 'practiced', 'rehearsed', 'repeated', 'reviewed', 'studied',
        
        'met', 'introduced', 'greeted', 'welcomed', 'invited', 'joined', 'participated', 'attended', 'gathered', 'assembled',
        'celebrated', 'congratulated', 'praised', 'complimented', 'thanked', 'apologized', 'forgave', 'excused', 'pardoned', 'blamed',
        'accused', 'criticized', 'scolded', 'punished', 'rewarded', 'encouraged', 'supported', 'assisted', 'aided', 'cooperated',
        'collaborated', 'partnered', 'teamed', 'united', 'combined', 'merged', 'connected', 'linked', 'attached', 'fastened',
        'tied', 'bound', 'secured', 'protected', 'defended', 'guarded', 'shielded', 'covered', 'hidden', 'concealed',
        
        'enjoyed', 'appreciated', 'admired', 'respected', 'honored', 'valued', 'treasured', 'cherished', 'adored', 'worshipped',
        'impressed', 'amazed', 'astonished', 'surprised', 'shocked', 'stunned', 'frightened', 'scared', 'terrified', 'panicked',
        'worried', 'concerned', 'troubled', 'disturbed', 'upset', 'disappointed', 'frustrated', 'annoyed', 'irritated', 'angered',
        'enraged', 'furious', 'delighted', 'pleased', 'satisfied', 'content', 'happy', 'joyful', 'excited', 'thrilled',
        'calm', 'relaxed', 'peaceful', 'serene', 'comfortable', 'cozy', 'warm', 'cool', 'cold', 'hot',
        
        'educated', 'instructed', 'trained', 'coached', 'mentored', 'guided', 'directed', 'led', 'managed', 'supervised',
        'controlled', 'commanded', 'ordered', 'requested', 'demanded', 'required', 'needed', 'wanted', 'desired', 'wished',
        'craved', 'longed', 'yearned', 'missed', 'regretted', 'regretted', 'mourned', 'grieved', 'lamented', 'celebrated',
        'marked', 'commemorated', 'honored', 'observed', 'witnessed', 'experienced', 'encountered', 'faced', 'confronted', 'challenged',
        'overcame', 'conquered', 'defeated', 'won', 'lost', 'competed', 'contested', 'raced', 'battled', 'fought',
        
        'shopped', 'purchased', 'bought', 'sold', 'traded', 'exchanged', 'swapped', 'borrowed', 'lent', 'rented',
        'leased', 'owned', 'possessed', 'held', 'grasped', 'gripped', 'squeezed', 'pressed', 'crushed', 'cracked',
        'broke', 'shattered', 'damaged', 'destroyed', 'ruined', 'wrecked', 'demolished', 'repaired', 'fixed', 'mended',
        'restored', 'renewed', 'refreshed', 'updated', 'upgraded', 'downgraded', 'installed', 'uninstalled', 'downloaded', 'uploaded',
        'saved', 'stored', 'kept', 'preserved', 'maintained', 'sustained', 'supported', 'balanced', 'stabilized', 'steadied',
        
        'exercised', 'stretched', 'warmed', 'cooled', 'heated', 'chilled', 'froze', 'melted', 'boiled', 'steamed',
        'baked', 'roasted', 'grilled', 'fried', 'sauteed', 'seasoned', 'flavored', 'spiced', 'sweetened', 'salted',
        'served', 'presented', 'offered', 'provided', 'supplied', 'delivered', 'distributed', 'shared', 'divided', 'separated',
        'split', 'joined', 'combined', 'mixed', 'blended', 'stirred', 'shaken', 'poured', 'filled', 'emptied',
        'loaded', 'unloaded', 'shipped', 'transported', 'transferred', 'relocated', 'displaced', 'replaced', 'substituted', 'switched',
        
        'programmed', 'coded', 'developed', 'debugged', 'tested', 'deployed', 'launched', 'released', 'published', 'distributed',
        'marketed', 'advertised', 'promoted', 'sold', 'negotiated', 'bargained', 'agreed', 'disagreed', 'argued', 'debated',
        'discussed', 'consulted', 'advised', 'recommended', 'suggested', 'proposed', 'submitted', 'presented', 'demonstrated', 'showed',
        'exhibited', 'displayed', 'revealed', 'exposed', 'uncovered', 'discovered', 'invented', 'patented', 'copyrighted', 'trademarked',
        'licensed', 'authorized', 'approved', 'rejected', 'denied', 'refused', 'accepted', 'embraced', 'adopted', 'implemented',
        
        # Additional verbs - 200+ more
        'accelerated', 'accessed', 'accompanied', 'activated', 'actualized', 'adapted', 'addressed', 'administered', 'admitted', 'advocated',
        'affiliated', 'aggregated', 'aligned', 'allocated', 'amplified', 'analyzed', 'anchored', 'animated', 'annotated', 'anticipated',
        'appeared', 'applied', 'appointed', 'approached', 'approximated', 'archived', 'articulated', 'ascended', 'aspired', 'asserted',
        'assigned', 'associated', 'assumed', 'assured', 'attached', 'attacked', 'attained', 'attended', 'attracted', 'attributed',
        'audited', 'augmented', 'authenticated', 'automated', 'avoided', 'awarded', 'backed', 'badged', 'balanced', 'banned',
        'bargained', 'barked', 'based', 'bathed', 'battled', 'beamed', 'bearing', 'beaten', 'beckoned', 'behaved',
        'belonged', 'benefited', 'biased', 'bid', 'billed', 'binded', 'bit', 'blamed', 'blanked', 'blessed',
        'blocked', 'blogged', 'bloomed', 'blown', 'boarded', 'boasted', 'bolted', 'bombed', 'bonded', 'booked',
        'boosted', 'borrowed', 'bounced', 'bounded', 'boxed', 'braced', 'bragged', 'braided', 'braked', 'branded',
        'breached', 'breathed', 'bred', 'bridged', 'briefed', 'broadened', 'broadcasted', 'brushed', 'budgeted', 'buffered',
        'bumped', 'bundled', 'buried', 'burned', 'burst', 'bypassed', 'cached', 'calibrated', 'camped', 'canceled',
        'canvassed', 'captured', 'cared', 'cascaded', 'casted', 'catalogued', 'categorized', 'catered', 'caused', 'cautioned',
        'centered', 'certified', 'chained', 'challenged', 'championed', 'changed', 'channeled', 'charged', 'charmed', 'chased',
        'checked', 'cheered', 'cherished', 'chewed', 'chilled', 'chipped', 'chopped', 'circled', 'circulated', 'cited',
        'claimed', 'clarified', 'clashed', 'classified', 'cleaned', 'cleared', 'clicked', 'climbed', 'clocked', 'cloned',
        'closed', 'clustered', 'coached', 'coated', 'coded', 'collected', 'collided', 'colored', 'combined', 'commanded',
        'commented', 'committed', 'communicated', 'compared', 'competed', 'compiled', 'complained', 'completed', 'complied', 'composed',
        'compressed', 'computed', 'conceived', 'concentrated', 'conceptualized', 'concluded', 'condensed', 'conducted', 'configured', 'confirmed',
        'conflicted', 'conformed', 'confused', 'connected', 'conquered', 'consented', 'conserved', 'considered', 'consoled', 'consolidated',
        'constructed', 'consulted', 'consumed', 'contacted', 'contained', 'contemplated', 'contended', 'contested', 'continued', 'contracted',
        'contributed', 'controlled', 'converted', 'conveyed', 'cooked', 'cooled', 'cooperated', 'coordinated', 'copied', 'corrected',
        'correlated', 'corrupted', 'counted', 'coupled', 'covered', 'crafted', 'crashed', 'created', 'credited', 'criticized',
        'crossed', 'crowded', 'crushed', 'cultivated', 'cured', 'curved', 'customized', 'cycled', 'damaged', 'danced',
        'dared', 'darkened', 'dated', 'dazzled', 'dealt', 'debugged', 'deceived', 'decided', 'declared', 'declined',
        'decorated', 'decreased', 'dedicated', 'deducted', 'deepened', 'defeated', 'defended', 'defined', 'deflated', 'delayed',
        'delegated', 'deleted', 'deliberated', 'delivered', 'demanded', 'demonstrated', 'denied', 'departed', 'depended', 'depicted',
        'deployed', 'deposited', 'deprecated', 'derived', 'descended', 'described', 'designed', 'desired', 'destroyed', 'detailed',
        'detected', 'determined', 'developed', 'deviated', 'devised', 'diagnosed', 'differed', 'diffused', 'digitized', 'diluted',
        'diminished', 'directed', 'disabled', 'disagreed', 'disappeared', 'disappointed', 'discharged', 'disciplined', 'disclosed', 'disconnected',
        'discouraged', 'discovered', 'discussed', 'dismissed', 'displayed', 'disposed', 'disputed', 'dissected', 'dissolved', 'distinguished',
        'distributed', 'disturbed', 'dived', 'diverted', 'divided', 'documented', 'dominated', 'donated', 'doubled', 'doubted',
        'downloaded', 'drained', 'dramatized', 'dreamed', 'dressed', 'dried', 'drilled', 'driven', 'dropped', 'drummed',
        'duplicated', 'earned', 'echoed', 'edited', 'educated', 'elected', 'elevated', 'eliminated', 'embedded', 'emerged',
        'emphasized', 'employed', 'enabled', 'encoded', 'encountered', 'encouraged', 'ended', 'endorsed', 'enforced', 'engaged',
        'engineered', 'enhanced', 'enjoyed', 'enlarged', 'enlisted', 'ensured', 'entered', 'entertained', 'equipped', 'escaped',
        'established', 'estimated', 'evaluated', 'evaporated', 'evoked', 'evolved', 'examined', 'exceeded', 'excelled', 'exchanged',
        'excited', 'excluded', 'executed', 'exercised', 'exhibited', 'existed', 'expanded', 'expected', 'experienced', 'experimented',
        'explained', 'exploded', 'explored', 'exported', 'exposed', 'expressed', 'extended', 'extracted', 'fabricated', 'faced',
        'facilitated', 'factored', 'failed', 'fainted', 'fallen', 'familiarized', 'fascinated', 'fashioned', 'fastened', 'favored',
        'feared', 'featured', 'fed', 'felt', 'fetched', 'figured', 'filed', 'filled', 'filtered', 'finalized',
        'financed', 'finished', 'fired', 'fitted', 'fixed', 'flagged', 'flashed', 'flattened', 'flavored', 'flexed',
        'flipped', 'floated', 'flooded', 'flourished', 'flowed', 'fluctuated', 'flushed', 'focused', 'folded', 'followed',
        'forced', 'forecasted', 'formed', 'formatted', 'formulated', 'forwarded', 'founded', 'framed', 'freed', 'frozen',
        'frustrated', 'fulfilled', 'functioned', 'funded', 'furnished', 'gained', 'gathered', 'generated', 'gestured', 'given',
        'glowed', 'grabbed', 'graduated', 'granted', 'graphed', 'grasped', 'greeted', 'grew', 'grinded', 'gripped',
        'grouped', 'grown', 'guaranteed', 'guarded', 'guided', 'handled', 'happened', 'hardened', 'harmed', 'harvested',
        'headed', 'healed', 'heard', 'heated', 'helped', 'hidden', 'highlighted', 'hired', 'hit', 'hold',
        'honored', 'hoped', 'hosted', 'housed', 'hovered', 'hugged', 'hunted', 'hurried', 'hurt', 'identified',
        'ignored', 'illuminated', 'illustrated', 'imagined', 'imitated', 'immersed', 'impacted', 'implemented', 'implied', 'imported',
        'imposed', 'impressed', 'improved', 'included', 'incorporated', 'increased', 'indexed', 'indicated', 'induced', 'infected',
        'inferred', 'influenced', 'informed', 'inherited', 'initiated', 'injected', 'injured', 'innovated', 'input', 'inserted',
        'inspected', 'inspired', 'installed', 'instructed', 'insured', 'integrated', 'intended', 'interacted', 'intercepted', 'interested',
        'interfered', 'interpreted', 'interrupted', 'interviewed', 'introduced', 'invaded', 'invented', 'investigated', 'invested', 'invited',
        'involved', 'isolated', 'issued', 'iterated', 'joined', 'judged', 'jumped', 'justified', 'kept', 'kicked',
        'killed', 'kissed', 'knew', 'knocked', 'known', 'labeled', 'landed', 'lasted', 'laughed', 'launched',
        'layered', 'lead', 'leaked', 'learned', 'leased', 'left', 'lengthened', 'leveled', 'licensed', 'lifted',
        'lightened', 'liked', 'limited', 'linked', 'listed', 'listened', 'lived', 'loaded', 'located', 'locked',
        'logged', 'looked', 'looped', 'lost', 'loved', 'lowered', 'maintained', 'managed', 'manipulated', 'manufactured',
        'mapped', 'marked', 'marketed', 'married', 'masked', 'mastered', 'matched', 'materialized', 'maximized', 'meant',
        'measured', 'mediated', 'melted', 'memorized', 'mentioned', 'merged', 'met', 'migrated', 'minimized', 'missed',
        'mixed', 'modeled', 'modified', 'monitored', 'motivated', 'mounted', 'moved', 'multiplied', 'named', 'navigated',
        'needed', 'negotiated', 'networked', 'neutralized', 'nominated', 'normalized', 'noted', 'noticed', 'notified', 'numbered',
        'nursed', 'obeyed', 'objected', 'observed', 'obtained', 'occurred', 'offered', 'opened', 'operated', 'opposed',
        'optimized', 'ordered', 'organized', 'oriented', 'originated', 'outlined', 'output', 'overcome', 'overlapped', 'overlooked',
        'overridden', 'owned', 'packed', 'painted', 'paired', 'parked', 'participated', 'partnered', 'passed', 'patched',
        'paused', 'payed', 'penalized', 'perceived', 'performed', 'permitted', 'persisted', 'personalized', 'persuaded', 'photographed',
        'picked', 'pictured', 'pinned', 'placed', 'planned', 'planted', 'played', 'pleased', 'pledged', 'plotted',
        'pointed', 'polished', 'populated', 'positioned', 'possessed', 'posted', 'postponed', 'poured', 'powered', 'practiced',
        'praised', 'predicted', 'preferred', 'prepared', 'presented', 'preserved', 'pressed', 'prevented', 'priced', 'printed',
        'prioritized', 'processed', 'produced', 'programmed', 'projected', 'promised', 'promoted', 'prompted', 'proposed', 'protected',
        'protested', 'proved', 'provided', 'published', 'pulled', 'pumped', 'punched', 'purchased', 'pursued', 'pushed',
        'put', 'qualified', 'quantified', 'questioned', 'queued', 'quit', 'quoted', 'raced', 'raised', 'ran',
        'ranked', 'rated', 'reached', 'read', 'realized', 'reasoned', 'received', 'recognized', 'recommended', 'recorded',
        'recovered', 'recruited', 'reduced', 'referred', 'reflected', 'refreshed', 'refused', 'registered', 'regulated', 'rehearsed',
        'rejected', 'related', 'relaxed', 'released', 'relied', 'remained', 'remembered', 'reminded', 'removed', 'rendered',
        'repaired', 'repeated', 'replaced', 'replied', 'reported', 'represented', 'requested', 'required', 'rescued', 'researched',
        'reserved', 'reset', 'resided', 'resolved', 'responded', 'restored', 'restricted', 'resulted', 'retained', 'retired',
        'retrieved', 'returned', 'revealed', 'reviewed', 'revised', 'rewarded', 'ridden', 'risked', 'rotated', 'rounded',
        'routed', 'ruled', 'run', 'rushed', 'sacrificed', 'sailed', 'satisfied', 'saved', 'scaled', 'scanned',
        'scattered', 'scheduled', 'scored', 'screamed', 'searched', 'secured', 'selected', 'sent', 'separated', 'served',
        'set', 'settled', 'shared', 'shifted', 'shipped', 'shocked', 'shot', 'showed', 'shut', 'signed',
        'simplified', 'simulated', 'skipped', 'slept', 'sliced', 'slowed', 'smiled', 'smoothed', 'snapped', 'solved',
        'sorted', 'sounded', 'specialized', 'specified', 'spent', 'split', 'spoken', 'sponsored', 'spread', 'stabilized',
        'stacked', 'staffed', 'staged', 'standardized', 'started', 'stated', 'stayed', 'stepped', 'stimulated', 'stopped',
        'stored', 'strengthened', 'stressed', 'stretched', 'structured', 'struggled', 'stuck', 'studied', 'styled', 'submitted',
        'subscribed', 'succeeded', 'suffered', 'suggested', 'summarized', 'supervised', 'supplied', 'supported', 'supposed', 'surprised',
        'surrounded', 'survived', 'suspended', 'sustained', 'switched', 'symbolized', 'synchronized', 'synthesized', 'systematized', 'tackled',
        'tagged', 'taken', 'talked', 'targeted', 'tasted', 'taught', 'taxed', 'teamed', 'terminated', 'tested',
        'thanked', 'thawed', 'thought', 'threatened', 'threw', 'tightened', 'timed', 'titled', 'tolerated', 'topped',
        'touched', 'toured', 'tracked', 'traded', 'trained', 'transferred', 'transformed', 'translated', 'transmitted', 'transported',
        'trapped', 'traveled', 'treated', 'tried', 'triggered', 'trimmed', 'trusted', 'tuned', 'turned', 'typed',
        'understood', 'unified', 'updated', 'upgraded', 'uploaded', 'urged', 'used', 'utilized', 'validated', 'valued',
        'varied', 'verified', 'viewed', 'violated', 'visited', 'visualized', 'voiced', 'volunteered', 'voted', 'waited',
        'walked', 'wanted', 'warned', 'washed', 'watched', 'weighed', 'welcomed', 'went', 'widened', 'won',
        'wondered', 'worked', 'worried', 'wrapped', 'written', 'yelled', 'yielded', 'zoomed'
    ]

    # Dedup while preserving order
    seen = set()
    unique = []
    for v in verbs:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return unique[:max_count]


def load_tokens_override(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    try:
        tokens: List[str] = []
        with open(path, 'r') as f:
            for line in f:
                s = line.strip()
                if s:
                    tokens.append(s)
        # Dedup while preserving order
        seen = set()
        unique: List[str] = []
        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                unique.append(tok)
        return unique
    except Exception as e:
        print(f"Failed to load TOKENS_FILE={path}: {e}")
        return None


def select_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and t.cuda.is_available():
        return f'cuda:{GPU}'
    return 'cpu'


def load_lm_and_tokenizer(device: str) -> Tuple[LanguageModel, AutoTokenizer]:
    """
    Try a sequence of model identifiers/paths for nnsight LanguageModel and Transformers tokenizer.
    """
    tried: List[str] = []
    last_err: Optional[Exception] = None

    # Try configured lm first, then local repo gpt2, then HF ids
    candidates = [
        lm,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'gpt2')),
        'openai-community/gpt2',
        'gpt2',
    ]

    for cand in candidates:
        try:
            model = LanguageModel(cand, dispatch=True, device_map=device)
            tokenizer = AutoTokenizer.from_pretrained(cand)
            return model, tokenizer
        except Exception as e:
            tried.append(cand)
            last_err = e

    raise RuntimeError(f"Failed to load language model/tokenizer. Tried: {tried}. Last error: {last_err}")


def find_non_scale_checkpoint(root: str = None) -> Optional[str]:
    """Find a plausible non-Scale SAE checkpoint (ae.pt) under dictionaries/."""
    root = root or os.path.abspath(os.path.join(os.path.dirname(__file__), 'dictionaries'))
    for dirpath, dirnames, filenames in os.walk(root):
        if 'ae.pt' in filenames:
            return os.path.join(dirpath, 'ae.pt')
    return None


def load_sae_variant(
    device: str,
    variant: str,
    path: Optional[str],
    activation_dim: int = 768,
    dict_size: int = 32 * 768,
    k: int = 32,
    experts: int = 64,
    e: int = E,
    heaviside: bool = False,
):
    """
    variant: 'scale' or 'plain'
    Returns (ae, used_path)
    """
    used_path = path
    if used_path is None or not os.path.exists(used_path):
        if variant == 'scale':
            cand = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dictionaries', 'MultiEncoder_Scale', f'{LAYER}.pt'))
            if os.path.exists(cand):
                used_path = cand
        else:
            used_path = find_non_scale_checkpoint()
    if used_path is None or not os.path.exists(used_path):
        raise FileNotFoundError(f"Could not find checkpoint for variant={variant}. Tried path={path} and auto-discovery.")

    if variant == 'scale':
        ae = MultiExpertScaleAutoEncoder(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            experts=experts,
            e=e,
            heaviside=heaviside,
        )
    else:
        ae = MultiExpertAutoEncoder(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            experts=experts,
            e=e,
            heaviside=heaviside,
        )

    state = t.load(used_path, map_location=device)
    ae.load_state_dict(state)
    ae.to(device)
    ae.eval()
    return ae, used_path


def get_layer_module(model: LanguageModel, layer: int):
    return model.transformer.h[layer]


@t.no_grad()
def get_token_activation(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    layer_module,
    token_text: str,
    device: str,
) -> Optional[t.Tensor]:
    try:
        context_text = f"The word {token_text} is important."

        with model.trace(context_text, scan=False, validate=False):
            hidden_states = layer_module.output.save()

        acts = hidden_states
        if isinstance(acts, tuple):
            acts = acts[0]
        if acts.dim() == 3:
            acts = acts[0]

        toks = tokenizer.encode(context_text)
        target_ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(target_ids) == 1:
            target_id = target_ids[0]
            positions = [i for i, tid in enumerate(toks) if tid == target_id]
            if positions:
                pos = positions[-1]
                return acts[pos].to(device)

        # Fallback: middle position
        mid = len(toks) // 2
        return acts[mid].to(device)
    except Exception:
        return None


@t.no_grad()
def topn_features_and_expert(
    ae,
    x: t.Tensor,
    n: int,
) -> Tuple[Sequence[int], int]:
    """
    Compute top-n feature indices (global) and the top-1 expert id for activation x.
    Returns (indices_sorted_by_value_desc, expert_id).
    """
    # Ensure batch
    xb = x.unsqueeze(0) if x.dim() == 1 else x

    # Reproduce gating expert for top-1 expert id
    gate_logits = ae.gate(xb - ae.b_gate)
    gate_scores = t.softmax(gate_logits, dim=-1)
    expert_id = int(t.argmax(gate_scores, dim=-1).item())

    # Full feature vector
    f = ae.encode(xb)
    vals, idxs = f.topk(k=n, dim=-1, sorted=True)
    top_idxs = [int(i) for i in idxs[0].tolist() if float(vals[0][(idxs[0] == i).nonzero(as_tuple=True)[0][0]]) > 0]
    return top_idxs, expert_id


def histogram_overlap_counts(sets: List[set], n: int) -> List[int]:
    hist = [0] * (n + 1)
    m = len(sets)
    for i in range(m):
        si = sets[i]
        for j in range(i + 1, m):
            ov = len(si & sets[j])
            if 0 <= ov <= n:
                hist[ov] += 1
    return hist


def entropy_from_hist(hist: Sequence[int]) -> float:
    total = sum(hist)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in hist:
        if c > 0:
            p = c / total
            ent -= p * math.log(p + 1e-12)
    return float(ent)


def analyze():
    random.seed(SEED)
    t.manual_seed(SEED)

    ensure_dir(OUTPUT_ROOT)

    device = select_device()
    model, tokenizer = load_lm_and_tokenizer(device)
    # Load both SAEs
    ae_scale, path_scale = load_sae_variant(device, variant='scale', path=SCALE_MODEL_PATH)
    ae_plain, path_plain = load_sae_variant(device, variant='plain', path=PLAIN_MODEL_PATH)
    layer_module = get_layer_module(model, LAYER)

    # Token source: override file if provided, else default past verbs
    override_tokens = load_tokens_override(TOKENS_FILE)
    if override_tokens is not None:
        verbs = override_tokens
        print(f"Using {len(verbs)} tokens from TOKENS_FILE={TOKENS_FILE}.")
    else:
        verbs = load_past_verbs(max_count=NUM_PAST_VERBS)
        if len(verbs) < NUM_PAST_VERBS:
            print(f"Warning: only {len(verbs)} past verbs available; proceeding with fewer than requested {NUM_PAST_VERBS}.")

    # Collect per-verb data (shared activations, then branch into two SAEs)
    verb_ok: List[str] = []  # tokens that produced an activation
    acts_cache: List[Tuple[str, t.Tensor]] = []

    for i, v in enumerate(verbs):
        if (i % 100) == 0:
            print(f"Processing {i + 1}/{len(verbs)}: {v}")

        x = get_token_activation(model, tokenizer, layer_module, v, device)
        if x is None:
            continue
        acts_cache.append((v, x))
        verb_ok.append(v)

    # For each variant, compute top-n sets and experts
    def collect_for(ae_model):
        sets: List[set] = []
        experts: List[int] = []
        ok_tokens: List[str] = []
        for v, x in acts_cache:
            idxs, eid = topn_features_and_expert(ae_model, x, TOP_N_FEATURES)
            if len(idxs) == 0:
                continue
            sets.append(set(idxs))
            experts.append(eid)
            ok_tokens.append(v)
        return sets, experts, ok_tokens

    verb_sets_scale, verb_experts_scale, tokens_scale = collect_for(ae_scale)
    verb_sets_plain, verb_experts_plain, tokens_plain = collect_for(ae_plain)

    # Overall overlap distribution (per variant)
    overall_hist_scale = histogram_overlap_counts(verb_sets_scale, TOP_N_FEATURES)
    overall_hist_plain = histogram_overlap_counts(verb_sets_plain, TOP_N_FEATURES)

    # Per-expert grouping
    def per_expert_hists_from(sets: List[set], experts: List[int]):
        per_sets: Dict[int, List[set]] = defaultdict(list)
        for s, e in zip(sets, experts):
            per_sets[e].append(s)
        per_hists: Dict[int, List[int]] = {}
        for e, s_list in per_sets.items():
            per_hists[e] = histogram_overlap_counts(s_list, TOP_N_FEATURES)
        return per_sets, per_hists

    per_expert_sets_scale, per_expert_hists_scale = per_expert_hists_from(verb_sets_scale, verb_experts_scale)
    per_expert_sets_plain, per_expert_hists_plain = per_expert_hists_from(verb_sets_plain, verb_experts_plain)

    # Compute stats
    def summarize_hist(hist: List[int]) -> Dict[str, float]:
        total_pairs = sum(hist)
        peak = max(hist) if hist else 0
        peak_frac = (peak / total_pairs) if total_pairs else 0.0
        ent = entropy_from_hist(hist)
        return {
            'total_pairs': int(total_pairs),
            'peak_bin': int(hist.index(peak) if hist else 0),
            'peak_count': int(peak),
            'peak_fraction': float(peak_frac),
            'entropy_nats': float(ent),
        }

    summary = {
        'config': {
            'layer': LAYER,
            'top_n': TOP_N_FEATURES,
            'num_verbs_requested': NUM_PAST_VERBS,
            'num_verbs_activations': len(verb_ok),
            'device': device,
            'lm_path': lm,
            'scale_path': path_scale,
            'plain_path': path_plain,
        },
        'scale': {
            'overall': {'hist': overall_hist_scale, **summarize_hist(overall_hist_scale)},
            'per_expert': {},
            'num_tokens_used': len(tokens_scale),
        },
        'plain': {
            'overall': {'hist': overall_hist_plain, **summarize_hist(overall_hist_plain)},
            'per_expert': {},
            'num_tokens_used': len(tokens_plain),
        },
    }

    for e, hist in per_expert_hists_scale.items():
        summary['scale']['per_expert'][str(e)] = {
            'hist': hist,
            **summarize_hist(hist),
            'num_verbs_in_expert': len(per_expert_sets_scale[e]),
        }
    for e, hist in per_expert_hists_plain.items():
        summary['plain']['per_expert'][str(e)] = {
            'hist': hist,
            **summarize_hist(hist),
            'num_verbs_in_expert': len(per_expert_sets_plain[e]),
        }

    # Also compute frequency of exact top-n sets per expert for peakedness
    per_expert_combo_freq_scale: Dict[str, List[Tuple[str, int]]] = {}
    for e, sets in per_expert_sets_scale.items():
        cnt = Counter(tuple(sorted(list(s))) for s in sets)
        top5 = cnt.most_common(5)
        per_expert_combo_freq_scale[str(e)] = [("-".join(map(str, k)), v) for k, v in top5]
    summary['scale']['per_expert_combo_top5'] = per_expert_combo_freq_scale

    per_expert_combo_freq_plain: Dict[str, List[Tuple[str, int]]] = {}
    for e, sets in per_expert_sets_plain.items():
        cnt = Counter(tuple(sorted(list(s))) for s in sets)
        top5 = cnt.most_common(5)
        per_expert_combo_freq_plain[str(e)] = [("-".join(map(str, k)), v) for k, v in top5]
    summary['plain']['per_expert_combo_top5'] = per_expert_combo_freq_plain

    # ------------------------------------------------------------
    # Feature-level partner stats (co-occurrence graph on top-N)
    # ------------------------------------------------------------
    def feature_partner_stats(sets: List[set]):
        feature_token_count: Counter[int] = Counter()
        partner_counts: Dict[int, Counter[int]] = defaultdict(Counter)
        for s in sets:
            for f_id in s:
                feature_token_count[f_id] += 1
                for g_id in s:
                    if g_id != f_id:
                        partner_counts[f_id][g_id] += 1
        return feature_token_count, partner_counts

    feature_token_count_scale, partner_counts_scale = feature_partner_stats(verb_sets_scale)
    feature_token_count_plain, partner_counts_plain = feature_partner_stats(verb_sets_plain)

    def entropy_counts(counter: Counter[int]) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0
        ent = 0.0
        for c in counter.values():
            if c > 0:
                p = c / total
                ent -= p * math.log(p + 1e-12)
        return float(ent)

    def build_feature_rows(feature_token_count: Counter[int], partner_counts: Dict[int, Counter[int]]):
        rows: List[Dict[str, float | int]] = []
        for f_id, freq in feature_token_count.items():
            if freq < MIN_FEATURE_COUNT:
                continue
            partners = partner_counts.get(f_id, Counter())
            total_co = sum(partners.values())
            num_partners = len(partners)
            if total_co > 0 and num_partners > 0:
                max_partner = max(partners.values())
                max_share = max_partner / total_co
                ent = entropy_counts(partners)
            else:
                max_share = 0.0
                ent = 0.0
            rows.append({
                'feature_id': int(f_id),
                'tokens_with_feature': int(freq),
                'num_partners': int(num_partners),
                'cooccurrence_total': int(total_co),
                'max_partner_share': float(max_share),
                'partner_entropy_nats': float(ent),
            })
        return rows

    feature_rows_scale = build_feature_rows(feature_token_count_scale, partner_counts_scale)
    feature_rows_plain = build_feature_rows(feature_token_count_plain, partner_counts_plain)

    # Save feature stats CSV
    feat_csv_path_scale = os.path.join(OUTPUT_ROOT, 'feature_partner_stats_Scale.csv')
    with open(feat_csv_path_scale, 'w', newline='') as fcsv:
        fieldnames = list(feature_rows_scale[0].keys()) if feature_rows_scale else ['feature_id','tokens_with_feature','num_partners','cooccurrence_total','max_partner_share','partner_entropy_nats']
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_rows_scale:
            writer.writerow(row)

    feat_csv_path_plain = os.path.join(OUTPUT_ROOT, 'feature_partner_stats_Plain.csv')
    with open(feat_csv_path_plain, 'w', newline='') as fcsv:
        fieldnames = list(feature_rows_plain[0].keys()) if feature_rows_plain else ['feature_id','tokens_with_feature','num_partners','cooccurrence_total','max_partner_share','partner_entropy_nats']
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_rows_plain:
            writer.writerow(row)

    # Save expert summary CSV (compact, not per-expert plots)
    exp_csv_path_scale = os.path.join(OUTPUT_ROOT, 'expert_overlap_summary_Scale.csv')
    with open(exp_csv_path_scale, 'w', newline='') as ecsv:
        writer = csv.writer(ecsv)
        writer.writerow(['expert_id', 'num_verbs_in_expert', 'total_pairs', 'peak_bin', 'peak_count', 'peak_fraction', 'entropy_nats'])
        for e, hist in per_expert_hists_scale.items():
            s = summarize_hist(hist)
            writer.writerow([e, len(per_expert_sets_scale[e]), s['total_pairs'], s['peak_bin'], s['peak_count'], s['peak_fraction'], s['entropy_nats']])

    exp_csv_path_plain = os.path.join(OUTPUT_ROOT, 'expert_overlap_summary_Plain.csv')
    with open(exp_csv_path_plain, 'w', newline='') as ecsv:
        writer = csv.writer(ecsv)
        writer.writerow(['expert_id', 'num_verbs_in_expert', 'total_pairs', 'peak_bin', 'peak_count', 'peak_fraction', 'entropy_nats'])
        for e, hist in per_expert_hists_plain.items():
            s = summarize_hist(hist)
            writer.writerow([e, len(per_expert_sets_plain[e]), s['total_pairs'], s['peak_bin'], s['peak_count'], s['peak_fraction'], s['entropy_nats']])

    # Save JSON
    with open(os.path.join(OUTPUT_ROOT, 'feature_combo_overlap_summary_both.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Plotting (optional): save simple bar plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        def plot_hist_overlay(hist_a: List[int], hist_b: List[int], labels: Tuple[str, str], title: str, path: str):
            # Overlapping bar histograms (same centers) with transparency
            xs = list(range(max(len(hist_a), len(hist_b))))
            ya = hist_a + [0] * (len(xs) - len(hist_a))
            yb = hist_b + [0] * (len(xs) - len(hist_b))

            # Normalize to proportions (each distribution sums to 1)
            sa = float(sum(ya))
            sb = float(sum(yb))
            pa = [y / sa if sa > 0 else 0.0 for y in ya]
            pb = [y / sb if sb > 0 else 0.0 for y in yb]
            bw = 0.65

            plt.figure(figsize=(6.8, 4.4))
            plt.bar(xs, pa, width=bw, color='tab:blue', alpha=0.5, label=labels[0], align='center')
            plt.bar(xs, pb, width=bw, color='tab:orange', alpha=0.5, label=labels[1], align='center')
            plt.xticks(xs, xs)
            plt.xlabel(f'Overlap count (0..{TOP_N_FEATURES})')
            plt.ylabel('Proportion of pairs')
            plt.title(title)
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path, dpi=200)
            plt.close()

        if PLOT_OVERALL:
            plot_hist_overlay(
                overall_hist_scale,
                overall_hist_plain,
                labels=("Scale", "Plain"),
                title=f'Overall top-{TOP_N_FEATURES} overlap distribution (both)',
                path=os.path.join(OUTPUT_ROOT, f'overall_overlap_top{TOP_N_FEATURES}_both.png'),
            )

            # ECDF overlay of Jaccard (overlap/TOP_N_FEATURES)
            def ecdf_from_hist(hist: List[int]):
                total_pairs = sum(hist)
                if total_pairs == 0:
                    return [k / TOP_N_FEATURES for k in range(len(hist))], [0.0] * len(hist)
                xs = [k / TOP_N_FEATURES for k in range(len(hist))]
                ps = [c / total_pairs for c in hist]
                cdf = []
                acc = 0.0
                for p in ps:
                    acc += p
                    cdf.append(acc)
                return xs, cdf

            xs_a, cdf_a = ecdf_from_hist(overall_hist_scale)
            xs_b, cdf_b = ecdf_from_hist(overall_hist_plain)
            plt.figure(figsize=(6.5, 4.2))
            plt.step(xs_a, cdf_a, where='post', label='Scale')
            plt.step(xs_b, cdf_b, where='post', label='Plain')
            plt.xlabel('Jaccard similarity (overlap / topN)')
            plt.ylabel('ECDF')
            plt.title(f'Overall ECDF of top-{TOP_N_FEATURES} set similarity (both)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_ROOT, f'overall_ecdf_top{TOP_N_FEATURES}_both.png'), dpi=200)
            plt.close()

        if PLOT_PER_EXPERT:
            # Note: Expert IDs are separate per SAE; we only produce per-SAE CSVs above.
            pass

        if PLOT_FEATURE_SCATTER and (feature_rows_scale or feature_rows_plain):
            # scatter: x=partner_entropy, y=tokens_with_feature (log scale-ish), size ~ num_partners
            plt.figure(figsize=(7.2,5.0))
            if feature_rows_scale:
                xs = [row['partner_entropy_nats'] for row in feature_rows_scale]
                ys = [row['tokens_with_feature'] for row in feature_rows_scale]
                ss = [max(10, 10 * math.sqrt(row['num_partners'])) for row in feature_rows_scale]
                plt.scatter(xs, ys, s=ss, alpha=0.35, c='tab:blue', edgecolors='none', label='Scale')
            if feature_rows_plain:
                xs = [row['partner_entropy_nats'] for row in feature_rows_plain]
                ys = [row['tokens_with_feature'] for row in feature_rows_plain]
                ss = [max(10, 10 * math.sqrt(row['num_partners'])) for row in feature_rows_plain]
                plt.scatter(xs, ys, s=ss, alpha=0.35, c='tab:orange', edgecolors='none', label='Plain')
            plt.xlabel('Partner entropy (nats)')
            plt.ylabel('Tokens with feature')
            plt.yscale('log')
            plt.title('Feature co-occurrence concentration vs frequency (both)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_ROOT, 'feature_entropy_vs_freq_both.png'), dpi=200)
            plt.close()

    except Exception as e:
        print(f"Plotting skipped due to error: {e}")

    # Save a TSV of verb -> expert and features for debugging
    with open(os.path.join(OUTPUT_ROOT, 'verb_expert_assignments_Scale.tsv'), 'w') as f:
        f.write('verb\texpert_id\n')
        for v, e in zip(tokens_scale, verb_experts_scale):
            f.write(f"{v}\t{e}\n")
    with open(os.path.join(OUTPUT_ROOT, 'verb_expert_assignments_Plain.tsv'), 'w') as f:
        f.write('verb\texpert_id\n')
        for v, e in zip(tokens_plain, verb_experts_plain):
            f.write(f"{v}\t{e}\n")

    # ------------------------------------------------------------
    # Tokens with larger Scale vs Plain differences
    #   - Compare per-token top-N sets across variants via Jaccard
    #   - Also record expert mismatches
    #   - Export ranked lists for quick reuse as new token arrays
    # ------------------------------------------------------------
    try:
        # Build token->(set, expert) maps
        tok2set_scale = {tok: s for tok, s in zip(tokens_scale, verb_sets_scale)}
        tok2set_plain = {tok: s for tok, s in zip(tokens_plain, verb_sets_plain)}
        tok2exp_scale = {tok: e for tok, e in zip(tokens_scale, verb_experts_scale)}
        tok2exp_plain = {tok: e for tok, e in zip(tokens_plain, verb_experts_plain)}

        common_tokens = sorted(set(tok2set_scale.keys()) & set(tok2set_plain.keys()))
        diff_rows = []
        for tok in common_tokens:
            sA = tok2set_scale.get(tok, set())
            sB = tok2set_plain.get(tok, set())
            inter = len(sA & sB)
            union = len(sA | sB)
            j = (inter / union) if union > 0 else 0.0
            eA = tok2exp_scale.get(tok, -1)
            eB = tok2exp_plain.get(tok, -1)
            diff_rows.append({
                'token': tok,
                'overlap': inter,
                'union': union,
                'jaccard': j,
                'expert_scale': eA,
                'expert_plain': eB,
                'expert_mismatch': int(eA != eB),
                'scale_indices': '-'.join(map(str, sorted(list(sA)))),
                'plain_indices': '-'.join(map(str, sorted(list(sB)))),
            })

        # Sort by ascending Jaccard (most different first)
        diff_rows.sort(key=lambda r: (r['jaccard'], -r['expert_mismatch']))

        # Write full TSV
        diff_tsv = os.path.join(OUTPUT_ROOT, 'tokens_variant_difference.tsv')
        with open(diff_tsv, 'w') as f:
            f.write('token\tjaccard\toverlap\tunion\texpert_scale\texpert_plain\texpert_mismatch\tscale_indices\tplain_indices\n')
            for r in diff_rows:
                f.write(f"{r['token']}\t{r['jaccard']:.6f}\t{r['overlap']}\t{r['union']}\t{r['expert_scale']}\t{r['expert_plain']}\t{r['expert_mismatch']}\t{r['scale_indices']}\t{r['plain_indices']}\n")

        # Write top-K most different tokens (lowest Jaccard)
        topk_tokens = [r['token'] for r in diff_rows[:DIFF_TOP_K]]
        with open(os.path.join(OUTPUT_ROOT, f'tokens_low_jaccard_top{DIFF_TOP_K}.txt'), 'w') as f:
            for tok in topk_tokens:
                f.write(tok + '\n')

        # Write all tokens below threshold
        thr_tokens = [r['token'] for r in diff_rows if r['jaccard'] <= JACCARD_THRESHOLD]
        with open(os.path.join(OUTPUT_ROOT, f'tokens_low_jaccard_le_{JACCARD_THRESHOLD:.2f}.txt'), 'w') as f:
            for tok in thr_tokens:
                f.write(tok + '\n')

        # Expert mismatch list
        mismatch_tokens = [r['token'] for r in diff_rows if r['expert_mismatch'] == 1]
        with open(os.path.join(OUTPUT_ROOT, 'tokens_expert_mismatch.txt'), 'w') as f:
            for tok in mismatch_tokens:
                f.write(tok + '\n')

    except Exception as e:
        print(f"Token-difference export skipped due to error: {e}")

    # ------------------------------------------------------------
    # Encoder matrix energy split (Low-frequency vs High-frequency)
    #   - Scale: use ae_scale.decompose_low_high on encoder weights
    #   - Plain: LP = per-expert row-mean repeated; HP = residual
    #   - Metrics per expert: E_lp, E_hp, hp_frac; for Scale also hp_frac_scaled
    #   - Outputs: CSV per variant + JSON overview + plots
    # ------------------------------------------------------------
    try:
        def frob2(x: t.Tensor) -> float:
            return float((x.float()**2).sum().item())

        # Collect per-expert energy for Scale
        scale_rows = []
        E_lp_sum_s = 0.0
        E_hp_sum_s = 0.0
        E_hp_scaled_sum_s = 0.0
        for ei, expert in enumerate(ae_scale.expert_modules):
            M = expert.encoder.weight.data  # [rows=expert_dict_size, cols=activation_dim]
            M_hat, A_LP, A_HP = ae_scale.decompose_low_high(M, ae_scale.omega[ei], position="encoder")
            E_lp = frob2(A_LP)
            E_hp = frob2(A_HP)
            scale_factor = float(ae_scale.omega[ei].item() + 1.0)
            E_hp_scaled = frob2(A_HP * scale_factor)
            denom = E_lp + E_hp
            hp_frac = (E_hp / denom) if denom > 0 else 0.0
            denom_scaled = E_lp + E_hp_scaled
            hp_frac_scaled = (E_hp_scaled / denom_scaled) if denom_scaled > 0 else 0.0
            scale_rows.append({
                'expert_id': ei,
                'E_lp': E_lp,
                'E_hp': E_hp,
                'hp_frac': hp_frac,
                'E_hp_scaled': E_hp_scaled,
                'hp_frac_scaled': hp_frac_scaled,
            })
            E_lp_sum_s += E_lp
            E_hp_sum_s += E_hp
            E_hp_scaled_sum_s += E_hp_scaled

        # Collect per-expert energy for Plain
        plain_rows = []
        E_lp_sum_p = 0.0
        E_hp_sum_p = 0.0
        for ei, expert in enumerate(ae_plain.expert_modules):
            M = expert.encoder.weight.data
            expert_avg = M.mean(dim=0, keepdim=True)               # [1, in_dim]
            A_LP = expert_avg.expand_as(M).contiguous()            # repeat along rows
            A_HP = (M - A_LP)
            E_lp = frob2(A_LP)
            E_hp = frob2(A_HP)
            denom = E_lp + E_hp
            hp_frac = (E_hp / denom) if denom > 0 else 0.0
            plain_rows.append({
                'expert_id': ei,
                'E_lp': E_lp,
                'E_hp': E_hp,
                'hp_frac': hp_frac,
            })
            E_lp_sum_p += E_lp
            E_hp_sum_p += E_hp

        # Write CSVs
        with open(os.path.join(OUTPUT_ROOT, 'encoder_energy_summary_Scale.csv'), 'w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['expert_id', 'E_lp', 'E_hp', 'hp_frac', 'E_hp_scaled', 'hp_frac_scaled'])
            for r in scale_rows:
                writer.writerow([r['expert_id'], r['E_lp'], r['E_hp'], r['hp_frac'], r['E_hp_scaled'], r['hp_frac_scaled']])
        with open(os.path.join(OUTPUT_ROOT, 'encoder_energy_summary_Plain.csv'), 'w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['expert_id', 'E_lp', 'E_hp', 'hp_frac'])
            for r in plain_rows:
                writer.writerow([r['expert_id'], r['E_lp'], r['E_hp'], r['hp_frac']])

        # JSON overview
        energy_overview = {
            'scale_totals': {
                'E_lp_total': E_lp_sum_s,
                'E_hp_total': E_hp_sum_s,
                'E_hp_scaled_total': E_hp_scaled_sum_s,
                'hp_frac_total_raw': (E_hp_sum_s / (E_lp_sum_s + E_hp_sum_s)) if (E_lp_sum_s + E_hp_sum_s) > 0 else 0.0,
                'hp_frac_total_scaled': (E_hp_scaled_sum_s / (E_lp_sum_s + E_hp_scaled_sum_s)) if (E_lp_sum_s + E_hp_scaled_sum_s) > 0 else 0.0,
            },
            'plain_totals': {
                'E_lp_total': E_lp_sum_p,
                'E_hp_total': E_hp_sum_p,
                'hp_frac_total_raw': (E_hp_sum_p / (E_lp_sum_p + E_hp_sum_p)) if (E_lp_sum_p + E_hp_sum_p) > 0 else 0.0,
            },
        }
        with open(os.path.join(OUTPUT_ROOT, 'encoder_energy_overview.json'), 'w') as f:
            json.dump(energy_overview, f, indent=2)

        # Plots
        try:
            import matplotlib.pyplot as plt

            # Histogram overlay of per-expert HP fraction (raw) for both variants
            hp_s = [r['hp_frac'] for r in scale_rows]
            hp_p = [r['hp_frac'] for r in plain_rows]
            bins = [i/20 for i in range(21)]  # 0..1 in 0.05 steps
            plt.figure(figsize=(6.8, 4.4))
            plt.hist(hp_s, bins=bins, density=True, alpha=0.5, label='Scale', color='tab:blue')
            plt.hist(hp_p, bins=bins, density=True, alpha=0.5, label='Plain', color='tab:orange')
            plt.xlabel('High-frequency energy fraction (per expert)')
            plt.ylabel('Density')
            plt.title('Encoder HP fraction per expert (raw)')
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_ROOT, 'encoder_hp_fraction_hist_both.png'), dpi=200)
            plt.close()

            # Total energy split stacked bars (raw HP)
            def stacked_bar(lp, hp, title, path):
                total = lp + hp
                p_lp = lp / total if total > 0 else 0.0
                p_hp = hp / total if total > 0 else 0.0
                xs = [0, 1]
                labels = ['Scale', 'Plain']
                lp_vals = [p_lp, (E_lp_sum_p/(E_lp_sum_p+E_hp_sum_p)) if (E_lp_sum_p+E_hp_sum_p)>0 else 0.0]
                hp_vals = [p_hp, (E_hp_sum_p/(E_lp_sum_p+E_hp_sum_p)) if (E_lp_sum_p+E_hp_sum_p)>0 else 0.0]
                plt.figure(figsize=(5.4, 4.4))
                plt.bar(xs, lp_vals, color='tab:green', alpha=0.7, label='Low-freq')
                plt.bar(xs, hp_vals, bottom=lp_vals, color='tab:red', alpha=0.7, label='High-freq')
                plt.xticks(xs, labels)
                plt.ylabel('Proportion of total energy')
                plt.title(title)
                plt.ylim(0, 1)
                plt.grid(True, axis='y', alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(path, dpi=200)
                plt.close()

            stacked_bar(E_lp_sum_s, E_hp_sum_s,
                        title='Encoder total energy split (raw)',
                        path=os.path.join(OUTPUT_ROOT, 'encoder_energy_total_split_raw.png'))

            # Total energy split for Scale using scaled HP
            total_s_scaled = E_lp_sum_s + E_hp_scaled_sum_s
            p_lp_s = E_lp_sum_s / total_s_scaled if total_s_scaled > 0 else 0.0
            p_hp_s = E_hp_scaled_sum_s / total_s_scaled if total_s_scaled > 0 else 0.0
            # For Plain, scaled == raw
            total_p = E_lp_sum_p + E_hp_sum_p
            p_lp_p = E_lp_sum_p / total_p if total_p > 0 else 0.0
            p_hp_p = E_hp_sum_p / total_p if total_p > 0 else 0.0
            plt.figure(figsize=(5.4, 4.4))
            xs = [0, 1]
            plt.bar(xs, [p_lp_s, p_lp_p], color='tab:green', alpha=0.7, label='Low-freq')
            plt.bar(xs, [p_hp_s, p_hp_p], bottom=[p_lp_s, p_lp_p], color='tab:red', alpha=0.7, label='High-freq')
            plt.xticks(xs, ['Scale(scaled HP)', 'Plain'])
            plt.ylabel('Proportion of total energy')
            plt.title('Encoder total energy split (Scale scaled HP)')
            plt.ylim(0, 1)
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_ROOT, 'encoder_energy_total_split_scaled.png'), dpi=200)
            plt.close()

        except Exception as pe:
            print(f"Plotting encoder energy split skipped due to error: {pe}")

    except Exception as e:
        print(f"Encoder energy analysis skipped due to error: {e}")


def main():
    analyze()


if __name__ == '__main__':
    main()
