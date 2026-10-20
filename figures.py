"""New reproducible figures; supplied source acquisition scripts are unchanged."""
from pathlib import Path
import os
import numpy as np
import pandas as pd

ORDER=['GPT-5.2','Claude Opus 4.8','Grok 4.3','DeepSeek V4 Pro','Qwen3-VL-235B-A22B-Instruct','Kimi K2.6']
LABELS={'Qwen3-VL-235B-A22B-Instruct':'Qwen3-VL-235B-A22B-Instruct'}

def make_figures(root:Path):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=root/'results/tables';out=root/'results/figures';out.mkdir(exist_ok=True)
    captions=[]
    main=pd.read_csv(d/'bootstrap_official_200.csv')
    subsets=[('01_FIR_all_200',main[main.exam=='FIR'],'FIR · 200 scored questions'),('02_BIR_all_200',main[main.exam=='BIR'],'BIR · 200 scored questions'),('03_FIR_text_only_180',pd.read_csv(d/'bootstrap_fir_text_only_180.csv'),'FIR · 180 text-only questions'),('04_FIR_image_containing_20',pd.read_csv(d/'bootstrap_fir_image_20.csv'),'FIR · 20 image-containing questions')]
    for filename,df,title in subsets:
        fig,ax=plt.subplots(figsize=(11,6))
        for strategy,offset,marker in [('item-wise',-.14,'o'),('whole-exam',.14,'s')]:
            g=df[df.strategy==strategy].set_index('model').reindex(ORDER)
            y=np.arange(len(ORDER))+offset;x=g.mean_accuracy_pct.to_numpy()
            ax.errorbar(x,y,xerr=np.vstack([x-g.acc_ci_low_pct.to_numpy(),g.acc_ci_high_pct.to_numpy()-x]),fmt=marker,capsize=4,label=strategy)
        labels=[x+(' [image-blind]' if x=='DeepSeek V4 Pro' and title.startswith('FIR') and 'text-only' not in title else '') for x in ORDER]
        ax.set_yticks(range(len(ORDER)),labels);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_xlabel('Correct answers (%) · mean of two replicates and 95% interval')
        ax.set_title(title,loc='left',pad=16);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2);ax.grid(axis='x',alpha=.2)
        fig.text(.02,.025,'10,000 item-cluster percentile bootstrap resamples; seed 2026. Descriptive comparisons.',fontsize=9)
        fig.subplots_adjust(left=.34,right=.97,top=.88,bottom=.26)
        for suffix in ['png','svg']:fig.savefig(out/f'{filename}.{suffix}',dpi=200)
        plt.close(fig)
        captions.append(f'{filename}: Mean accuracy and 95% percentile intervals from 10,000 question-cluster bootstrap resamples retaining both replicates (seed 2026). Accuracy axes run from 0% to 100%. No superiority/equivalence tests. DeepSeek received no images; its image-containing score is an operational image-blind result and not a visual-reasoning measurement. The text-only stratum does not imply that images elsewhere in the full-exam request were removed for the other models.')
    efficiency=pd.read_csv(d/'efficiency_by_condition.csv')
    i=5
    for exam in ['FIR','BIR']:
        for metric,label in [('final_attempt_latency_seconds','Sum of final-attempt API latency (seconds)'),('total_tokens','Total provider-reported tokens per 210-question run')]:
            fig,ax=plt.subplots(figsize=(11,6))
            df=efficiency[efficiency.exam==exam]
            for strategy,offset,marker in [('item-wise',-.14,'o'),('whole-exam',.14,'s')]:
                g=df[df.strategy==strategy].set_index('model').reindex(ORDER)
                ax.plot(g[metric],np.arange(len(ORDER))+offset,marker,linestyle='None',label=strategy)
            ax.set_yticks(range(len(ORDER)),ORDER);ax.invert_yaxis();ax.set_xlim(left=0)
            ax.set_xlabel(label);ax.set_title(f'{exam} · operational API metrics',loc='left',pad=16);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2);ax.grid(axis='x',alpha=.2)
            fig.text(.02,.025,'Mean of two runs. Final-attempt metrics exclude failed-attempt overhead; source-logged totals are retained separately.',fontsize=9)
            fig.subplots_adjust(left=.34,right=.97,top=.88,bottom=.26)
            filename=f'{i:02d}_{exam}_{"latency" if metric.startswith("final") else "tokens"}';i+=1
            for suffix in ['png','svg']:fig.savefig(out/f'{filename}.{suffix}',dpi=200)
            plt.close(fig)
            captions.append(f'{filename}: Mean of two retained runs. {label}. Serial item-wise requests are summed; whole-exam values refer to one request. No interpretation as intrinsic compute, energy, or isolated context-length effects. Final-attempt latency is separated from retry-inclusive/source-logged latency; Kimi BIR therefore differs from a source summary that included retry time. Estimated costs are plotted separately using the documented rate table.')
    (out/'FIGURE_CAPTIONS.txt').write_text('\n\n'.join(captions)+'\n',encoding='utf-8')


def make_cost_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.read_csv(root/'results/tables/efficiency_by_condition.csv')
    out=root/'results/figures'
    for number,exam in [(9,'FIR'),(10,'BIR')]:
        fig,ax=plt.subplots(figsize=(11,6))
        for strategy,offset,marker in [('item-wise',-.14,'o'),('whole-exam',.14,'s')]:
            g=df[(df.exam==exam)&(df.strategy==strategy)].set_index('model').reindex(ORDER)
            ax.plot(g.estimated_api_cost_usd,np.arange(len(ORDER))+offset,marker,linestyle='None',label=strategy)
        ax.set_yticks(range(len(ORDER)),ORDER);ax.invert_yaxis();ax.set_xlim(left=0)
        ax.set_xlabel('Estimated API cost per 210-question run (USD)');ax.set_title(exam+' · estimated API cost',loc='left')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2);ax.grid(axis='x',alpha=.2)
        fig.subplots_adjust(left=.34,right=.97,top=.88,bottom=.26)
        fig.text(.02,.025,'Mean of two runs. Standard input/output rates; no cache discounts. Rate sources: config/api_prices.csv.',fontsize=9)
        for suffix in ['png','svg']:fig.savefig(out/f'{number:02d}_{exam}_cost.{suffix}',dpi=200)
        plt.close(fig)
