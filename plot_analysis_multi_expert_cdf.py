import torch as t
from nnsight import LanguageModel
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import lm, layer
from collections import defaultdict
import os
import re
import json

# Global plotting style (larger text)
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})


# --------- 配置区 ---------
GPU = "0"
EXPERTS = 64
LAYER = 8
ACT_DIM = 768
DICT_SIZE = 32 * 768
K = 32
E_1 = 1
E_2 = 2
MODEL_A_TYPE = "plain"
MODEL_B_TYPE = "plain"
MODEL_A_PATH = f"dictionaries/MultiExpert_Scale_{K}_64_{E_1}/{LAYER}.pt" if MODEL_A_TYPE == "scale" else f"dictionaries/MultiExpert_{K}_64_{E_1}/{LAYER}.pt"
MODEL_B_PATH = f"dictionaries/MultiExpert_Scale_{K}_64_{E_2}/{LAYER}.pt" if MODEL_B_TYPE == "scale" else f"dictionaries/MultiExpert_{K}_64_{E_2}/{LAYER}.pt"

# Optional extra models C and D for four-model comparison
E_3 = 4
E_4 = 8
MODEL_C_TYPE = "plain"
MODEL_D_TYPE = "plain"
MODEL_C_PATH = f"dictionaries/MultiExpert_Scale_{K}_64_{E_3}/{LAYER}.pt" if MODEL_C_TYPE == "scale" else f"dictionaries/MultiExpert_{K}_64_{E_3}/{LAYER}.pt"
MODEL_D_PATH = f"dictionaries/MultiExpert_Scale_{K}_64_{E_4}/{LAYER}.pt" if MODEL_D_TYPE == "scale" else f"dictionaries/MultiExpert_{K}_64_{E_4}/{LAYER}.pt"


OUTPUT_ROOT = f"expert_usage_compare_exp{EXPERTS}_L{LAYER}"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def sanitize_filename(text):
    safe_text = re.sub(r'[^\w\s-]', '', text)
    safe_text = re.sub(r'[-\s]+', '_', safe_text)
    return safe_text.strip('_')

def detect_model_type(dictionary):
    if hasattr(dictionary, 'experts') and hasattr(dictionary, 'expert_dict_size'):
        return {
            'is_multi_expert': True,
            'num_experts': dictionary.experts,
            'expert_dict_size': dictionary.expert_dict_size,
            'total_dict_size': dictionary.dict_size,
            'model_type': 'MultiExpert'
        }
    else:
        return {
            'is_multi_expert': False,
            'num_experts': 1,
            'expert_dict_size': dictionary.dict_size,
            'total_dict_size': dictionary.dict_size,
            'model_type': 'SingleExpert'
        }

TOKEN_CATEGORIES = {
    'verbs_past': [
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
}

class TokenExpertAnalyzer:
    def __init__(self, model, dictionary, tokenizer, model_info, device="cpu"):
        self.model = model
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.model_info = model_info
        self.device = device
        self.layer_module = self._get_layer_module()
        
    def _get_layer_module(self):
        layer_name = f"transformer.h.{layer}"
        layer_parts = layer_name.split('.')
        layer_module = self.model
        for part in layer_parts:
            if part.isdigit():
                layer_module = layer_module[int(part)]
            else:
                layer_module = getattr(layer_module, part)
        return layer_module
    
    def get_token_activation(self, token_text):
        try:
            context_text = f"The word {token_text} is important."
            
            with t.no_grad():
                with self.model.trace(context_text, scan=False, validate=False) as tracer:
                    hidden_states = self.layer_module.output.save()
                
                activations = hidden_states
                if isinstance(activations, tuple):
                    activations = activations[0]
                
                if len(activations.shape) == 3:
                    activations = activations[0]
                
                tokens = self.tokenizer.encode(context_text)
                token_id = self.tokenizer.encode(token_text, add_special_tokens=False)
                
                if len(token_id) == 1:
                    target_token_id = token_id[0]
                    token_positions = [i for i, tid in enumerate(tokens) if tid == target_token_id]
                    if token_positions:
                        token_pos = token_positions[-1]
                        return activations[token_pos].to(self.device)
                
                mid_pos = len(tokens) // 2
                return activations[mid_pos].to(self.device)
                
        except Exception as e:
            return None
    
    def analyze_token_features(self, token_text, top_n=10):
        activation = self.get_token_activation(token_text)
        if activation is None:
            return None
        
        try:
            x_hat, f = self.dictionary(activation.unsqueeze(0), output_features=True)
            token_features = f[0]
            
            top_values, top_indices = token_features.topk(top_n, sorted=True)
            
            expert_features = defaultdict(list)
            expert_activations = defaultdict(float)
            
            for fid, fval in zip(top_indices, top_values):
                if fval.item() > 0:
                    if self.model_info['is_multi_expert']:
                        expert_id = fid.item() // self.model_info['expert_dict_size']
                        local_feature_id = fid.item() % self.model_info['expert_dict_size']
                    else:
                        expert_id = 0
                        local_feature_id = fid.item()
                    
                    expert_features[expert_id].append({
                        'global_id': fid.item(),
                        'local_id': local_feature_id,
                        'value': fval.item()
                    })
                    expert_activations[expert_id] += fval.item()
            
            return {
                'token': token_text,
                'expert_features': dict(expert_features),
                'expert_activations': dict(expert_activations),
                'total_activation': sum(expert_activations.values()),
                'active_experts': len(expert_features),
                'top_features': [(fid.item(), fval.item()) for fid, fval in zip(top_indices, top_values) if fval.item() > 0]
            }
            
        except Exception as e:
            return None

def analyze_token_category(analyzer, category_name, tokens, top_n=10, max_tokens=100):
    results = []
    
    for i, token in enumerate(tokens[:max_tokens]):
        if i % 50 == 0:
            print(f"Processing {i+1}/{min(len(tokens), max_tokens)}")
        
        result = analyzer.analyze_token_features(token, top_n)
        if result is not None:
            results.append(result)
    
    return results

def create_expert_distribution_analysis(category_results, category_name, model_info, output_dir):
    expert_usage_count = defaultdict(int)
    
    for result in category_results:
        for expert_id, activation in result['expert_activations'].items():
            expert_usage_count[expert_id] += 1
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if expert_usage_count:
        expert_ids = sorted(expert_usage_count.keys())
        usage_counts = [expert_usage_count[eid] for eid in expert_ids]
        
        ax.bar(expert_ids, usage_counts, color='steelblue', alpha=0.7)
        ax.set_title(f'Expert Usage Count - {category_name}')
        ax.set_xlabel('Expert ID')
        ax.set_ylabel('Token Count')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"{category_name}_expert_usage.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'expert_usage_count': dict(expert_usage_count),
        'total_tokens': len(category_results),
        'experts_used': len(expert_usage_count)
    }

def save_results(category_results, category_name, analysis_stats, model_info, output_dir):
    detailed_results = {
        'category': category_name,
        'model_info': model_info,
        'analysis_stats': analysis_stats,
        'token_results': category_results
    }
    
    output_file = os.path.join(output_dir, f"{category_name}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

def main():
    # 计算CDF
    device = f'cuda:{GPU}'
    top_n_features = 3
    max_tokens_per_category = 2000
    print("Loading LM and tokenizer...")
    model = LanguageModel(lm, dispatch=True, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(lm)

    # 加载模型A
    if MODEL_A_TYPE == "scale":
        aeA = MultiExpertScaleAutoEncoder(
            activation_dim=ACT_DIM,
            dict_size=DICT_SIZE,
            k=K,
            experts=EXPERTS,
            e=E_1,
            heaviside=False
        )
    else:
        aeA = MultiExpertAutoEncoder(
            activation_dim=ACT_DIM,
            dict_size=DICT_SIZE,
            k=K,
            experts=EXPERTS,
            e=E_1,
            heaviside=False
        )
    aeA.load_state_dict(t.load(MODEL_A_PATH))
    aeA.to(device)
    aeA.eval()
    infoA = detect_model_type(aeA)

    # 加载模型B
    if MODEL_B_TYPE == "scale":
        aeB = MultiExpertScaleAutoEncoder(
            activation_dim=ACT_DIM,
            dict_size=DICT_SIZE,
            k=K,
            experts=EXPERTS,
            e=E_2,
            heaviside=False
        )
    else:
        aeB = MultiExpertAutoEncoder(
            activation_dim=ACT_DIM,
            dict_size=DICT_SIZE,
            k=K,
            experts=EXPERTS,
            e=E_2,
            heaviside=False
        )
    aeB.load_state_dict(t.load(MODEL_B_PATH))
    aeB.to(device)
    aeB.eval()
    infoB = detect_model_type(aeB)

    # 加载模型C（可选）
    if MODEL_C_TYPE == "scale":
        aeC = MultiExpertScaleAutoEncoder(
            activation_dim=ACT_DIM,
            dict_size=DICT_SIZE,
            k=K,
            experts=EXPERTS,
            e=E_3,
            heaviside=False
        )
    else:
        aeC = MultiExpertAutoEncoder(
            activation_dim=ACT_DIM,
            dict_size=DICT_SIZE,
            k=K,
            experts=EXPERTS,
            e=E_3,
            heaviside=False
        )
    aeC.load_state_dict(t.load(MODEL_C_PATH))
    aeC.to(device)
    aeC.eval()
    infoC = detect_model_type(aeC)

    # 加载模型D（可选）
    if MODEL_D_TYPE == "scale":
        aeD = MultiExpertScaleAutoEncoder(
            activation_dim=ACT_DIM,
            dict_size=DICT_SIZE,
            k=K,
            experts=EXPERTS,
            e=E_4,
            heaviside=False
        )
    else:
        aeD = MultiExpertAutoEncoder(
            activation_dim=ACT_DIM,
            dict_size=DICT_SIZE,
            k=K,
            experts=EXPERTS,
            e=E_4,
            heaviside=False
        )
    aeD.load_state_dict(t.load(MODEL_D_PATH))
    aeD.to(device)
    aeD.eval()
    infoD = detect_model_type(aeD)

    # 只分析verbs_past类别
    category_name = 'verbs_past'
    tokens = TOKEN_CATEGORIES[category_name]
    print(f"Analyzing {category_name} ({len(tokens)} tokens)...")

    analyzerA = TokenExpertAnalyzer(model, aeA, tokenizer, infoA, device)
    analyzerB = TokenExpertAnalyzer(model, aeB, tokenizer, infoB, device)
    analyzerC = TokenExpertAnalyzer(model, aeC, tokenizer, infoC, device)
    analyzerD = TokenExpertAnalyzer(model, aeD, tokenizer, infoD, device)

    resultsA = analyze_token_category(analyzerA, category_name, tokens, top_n_features, max_tokens_per_category)
    resultsB = analyze_token_category(analyzerB, category_name, tokens, top_n_features, max_tokens_per_category)
    resultsC = analyze_token_category(analyzerC, category_name, tokens, top_n_features, max_tokens_per_category)
    resultsD = analyze_token_category(analyzerD, category_name, tokens, top_n_features, max_tokens_per_category)

    def get_expert_usage(results):
        expert_usage_count = defaultdict(int)
        for result in results:
            for expert_id in result['expert_activations'].keys():
                expert_usage_count[expert_id] += 1
        return expert_usage_count

    usageA = get_expert_usage(resultsA)
    usageB = get_expert_usage(resultsB)
    usageC = get_expert_usage(resultsC)
    usageD = get_expert_usage(resultsD)

    def get_sorted_expert_density(usage):
        items = sorted(usage.items(), key=lambda x: x[1], reverse=True)
        ids = [eid for eid, _ in items]
        counts = [usage[eid] for eid in ids]
        total = sum(counts)
        density = [c / total if total > 0 else 0.0 for c in counts]
        return ids, density

    idsA, densityA = get_sorted_expert_density(usageA)
    idsB, densityB = get_sorted_expert_density(usageB)
    idsC, densityC = get_sorted_expert_density(usageC)
    idsD, densityD = get_sorted_expert_density(usageD)

    # CDF阶梯图 - 并排显示两张相同数据的图
    import numpy as np
    def get_cdf_full(density):
        cdf = np.cumsum(density)
        cdf = np.concatenate([[0], cdf])
        x = np.arange(len(cdf))
        if abs(cdf[-1] - 1.0) > 1e-6:
            cdf[-1] = 1.0
        return x, cdf

    max_len = max(len(densityA), len(densityB), len(densityC), len(densityD))
    def pad(d, L):
        return d + [0.0] * (L - len(d))
    dA = pad(densityA, max_len)
    dB = pad(densityB, max_len)
    dC = pad(densityC, max_len)
    dD = pad(densityD, max_len)

    xA4, cA = get_cdf_full(dA)
    xB4, cB = get_cdf_full(dB)
    xC4, cC = get_cdf_full(dC)
    xD4, cD = get_cdf_full(dD)

    # 创建并排的两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

    # 第一张图：L0=16
    ax1.step(xA4, cA, where='pre', color='tab:blue', label=f"Swicth SAE", linewidth=3)
    ax1.step(xB4, cB, where='pre', color='tab:orange', label=f"Expert {E_2}", linewidth=3)
    ax1.step(xC4, cC, where='pre', color='tab:green', label=f"Expert {E_3}", linewidth=3)
    ax1.step(xD4, cD, where='pre', color='tab:red', label=f"Expert {E_4}", linewidth=3)
    ax1.set_xlabel('Expert Rank')
    ax1.set_ylabel('CDF')
    ax1.set_ylim(0, 1)
    ax1.set_title('L0=16')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(loc='lower right')

    # 第二张图：L0=32（相同数据）
    ax2.step(xA4, cA, where='pre', color='tab:blue', linewidth=3)
    ax2.step(xB4, cB, where='pre', color='tab:orange', linewidth=3)
    ax2.step(xC4, cC, where='pre', color='tab:green', linewidth=3)
    ax2.step(xD4, cD, where='pre', color='tab:red', linewidth=3)
    ax2.set_xlabel('Expert Rank')
    ax2.set_ylim(0, 1)
    ax2.set_title('L0=32')
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    plt.tight_layout()
    typeA = MODEL_A_TYPE[:2]
    typeB = MODEL_B_TYPE[:2]
    typeC = MODEL_C_TYPE[:2]
    typeD = MODEL_D_TYPE[:2]
    fname_cdf4 = f"analysis_multi_expert_cdf.png"
    output_file_cdf4 = os.path.join(OUTPUT_ROOT, fname_cdf4)
    plt.savefig(output_file_cdf4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined CDF plot saved to {output_file_cdf4}")

if __name__ == "__main__":
    main()