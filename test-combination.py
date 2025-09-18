import os
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch as t
from nnsight import LanguageModel
from transformers import AutoTokenizer

from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
from config import lm

GPU = "0"
LAYER = 8
import os as _os

E = int(_os.environ.get('E', '16'))
K = 128
SCALE_MODEL_PATH = f"dictionaries/MultiExpert_Scale_{K}_64_{E}/{LAYER}.pt"
PLAIN_MODEL_PATH = f"dictionaries/MultiExpert_{K}_64_{E}/{LAYER}.pt"

OUTPUT_ROOT = f"feature_combinations/K{K}_E{E}_L{LAYER}"

TOP_N_FEATURES = K
NUM_PAST_VERBS = 1500
SEED = 0

TOKENS_FILE = None

PLOT_OVERALL = True
PLOT_PER_EXPERT = False
PLOT_FEATURE_SCATTER = True

MIN_FEATURE_COUNT = 5
TOP_FEATURES_TO_LIST = 50

DIFF_TOP_K = 200
JACCARD_THRESHOLD = 0.25


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_past_verbs(max_count: int = 1500) -> List[str]:
    verbs: List[str] = []
    import importlib.util

    ta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test-activate.py'))
    if os.path.exists(ta_path):
        spec = importlib.util.spec_from_file_location("test_activate_mod", ta_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if hasattr(mod, 'TOKEN_CATEGORIES') and 'verbs_past' in mod.TOKEN_CATEGORIES:
            verbs = list(mod.TOKEN_CATEGORIES['verbs_past'])

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
        model = LanguageModel(cand, dispatch=True, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(cand)
        return model, tokenizer

    raise RuntimeError(f"Failed to load language model/tokenizer. Tried: {candidates}.")


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
    ae_scale, _ = load_sae_variant(device, variant='scale', path=SCALE_MODEL_PATH)
    ae_plain, _ = load_sae_variant(device, variant='plain', path=PLAIN_MODEL_PATH)
    layer_module = get_layer_module(model, LAYER)

    override_tokens = load_tokens_override(TOKENS_FILE)
    if override_tokens is not None:
        verbs = override_tokens
        print(f"Using {len(verbs)} tokens from TOKENS_FILE={TOKENS_FILE}.")
    else:
        verbs = load_past_verbs(max_count=NUM_PAST_VERBS)
        if len(verbs) < NUM_PAST_VERBS:
            print(f"Warning: only {len(verbs)} past verbs available; proceeding with fewer than requested {NUM_PAST_VERBS}.")

    verb_ok: List[str] = []
    acts_cache: List[Tuple[str, t.Tensor]] = []

    for i, v in enumerate(verbs):
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

    verb_sets_scale, _, _ = collect_for(ae_scale)
    verb_sets_plain, _, _ = collect_for(ae_plain)

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

    total_pairs_scale = float(sum(overall_hist_scale)) if sum(overall_hist_scale) > 0 else 0.0
    if total_pairs_scale > 0.0:
        avg_overlap_scale = float(sum(i * c for i, c in enumerate(overall_hist_scale))) / total_pairs_scale
    else:
        avg_overlap_scale = 0.0

    total_pairs_plain = float(sum(overall_hist_plain)) if sum(overall_hist_plain) > 0 else 0.0
    if total_pairs_plain > 0.0:
        avg_overlap_plain = float(sum(i * c for i, c in enumerate(overall_hist_plain))) / total_pairs_plain
    else:
        avg_overlap_plain = 0.0

    avg_similarity_scale = (avg_overlap_scale / float(TOP_N_FEATURES)) if TOP_N_FEATURES > 0 else 0.0
    avg_similarity_plain = (avg_overlap_plain / float(TOP_N_FEATURES)) if TOP_N_FEATURES > 0 else 0.0


    import matplotlib.pyplot as plt

    def plot_hist_overlay(hist_a: List[int], hist_b: List[int], labels: Tuple[str, str], title: str, path: str):
        xs = list(range(max(len(hist_a), len(hist_b))))
        ya = hist_a + [0] * (len(xs) - len(hist_a))
        yb = hist_b + [0] * (len(xs) - len(hist_b))

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
        pass

    if PLOT_FEATURE_SCATTER and (feature_rows_scale or feature_rows_plain):
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

    print("avg_similarity (scale):", avg_similarity_scale)
    print("avg_similarity (plain):", avg_similarity_plain)


def main():
    analyze()


if __name__ == '__main__':
    main()
