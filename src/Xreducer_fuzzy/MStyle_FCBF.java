package Xreducer_fuzzy;

import java.io.FileReader;
import java.util.Arrays;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import helpLib.Utils;
import helpLib.Utils_entropy;

public class MStyle_FCBF  extends MeasureStyle{
	public double m_lambda = 0.0;
	public boolean m_isMDL;
	public int m_bin;
	public MStyle_FCBF(Instances data, boolean isMDL, int bin, double lambda) throws Exception {
		super(data, null, null);
		// TODO Auto-generated constructor stub
		this.m_lambda = lambda;
		this.m_isMDL = isMDL;
		this.m_bin = bin;
		String lg = "";
		if(isMDL){
			lg = "-MDL";
		}
		else{
			lg = "-"+bin+"Bin";
		}
		this.algname =  "FCBF("+(this.m_lambda==-1.0?"log":(int)this.m_lambda)+")"+lg+"算法";
		this.m_selectAtt = getSelectedAtt();
		
	}
	public double getMeausureValue(boolean[] D, boolean[] B){
		return Utils_entropy.getSU(this.m_data, D, B);
	}
	 
	public boolean[] getReduceAtt(){
		int U = this.m_data.numInstances();
		int N = this.m_data.numAttributes();
		boolean[] newB = new boolean[N];
		boolean[] B = new boolean[N];
		boolean[] A = Utils.Instances2FullBoolean(this.m_data);
		boolean[] D = Utils.Instances2DecBoolean(this.m_data);
		int[] cfbf_index = new int[N-1]; //不包括决策属性
		double[] cfbf_value = new double[N-1];
		
		long time_start = Utils.getCurrenttime();		
		//离散化
		if(this.m_isMDL){
			 weka.filters.supervised.attribute.Discretize sd = new weka.filters.supervised.attribute.Discretize();
			try {
				sd.setInputFormat(this.m_data);
				this.m_data = Filter.useFilter(this.m_data , sd);
			} catch (Exception e) {
					// TODO Auto-generated catch block
				e.printStackTrace();
			}		 
		}
		else{
		 
			weka.filters.unsupervised.attribute.Discretize unsd = new weka.filters.unsupervised.attribute.Discretize();
			unsd.setBins(this.m_bin);
			//unsd.setUseEqualFrequency(true); // If set to true, equal-frequency binning will be used instead of equal-width binning.
			try {
				unsd.setInputFormat(this.m_data);
				this.m_data = Filter.useFilter(this.m_data , unsd);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
		
		for(int k=0;k<N-1;++k){
			B[k]=true;
			cfbf_value[k]=this.getMeausureValue(D, B);
			cfbf_index[k]=k;
			B[k]=false;
		}
		if(this.m_lambda==-1)
		{
			int log = (int) (N/Utils.log2(N));
			this.m_lambda = cfbf_value[log];
		}
		//排序
		double temp;
		int tempindex;
		for(int i=0;i<N-1;++i){/* 冒泡法排序 */ 
			for(int j=0;j< N-i-2;++j){
				if(cfbf_value[j]<cfbf_value[j+1]) {
					//交换cfbf_value
					temp = cfbf_value[j];
					cfbf_value[j] = cfbf_value[j + 1];
					cfbf_value[j + 1] = temp;
					//交换cfbf_index
					tempindex = cfbf_index[j];
					cfbf_index[j] = cfbf_index[j + 1];
					cfbf_index[j + 1] = tempindex;
				}
			}
		}
		//System.out.println(Arrays.toString(cfbf_value));
		newB[cfbf_index[0]]=true;
	 
		int cnt = 1;
		while(cnt!=(N-1) && cfbf_value[cnt]>=this.m_lambda){
			boolean[] X = new boolean[N];
			X[cfbf_index[cnt]]=true;//newone
			boolean isRud = false;
			for(int i=0;i<cnt;++i){
				if(newB[cfbf_index[i]]){
					boolean[] Y = new boolean[N];
					Y[cfbf_index[i]]=true;//oldone			 
					//if(this.getMeausureValue(X, Y)>=this.m_lambda*cfbf_value[cnt]){//不相关	
					if(this.getMeausureValue(X, Y)>=cfbf_value[cnt]){//不相关
						isRud = true;
						break;
					}
					
				}
			}
			if(!isRud){
				newB[cfbf_index[cnt]]=true;
				//System.out.println(Arrays.toString(Utils.boolean2select(newB))+"->"+cfbf_index[cnt]+":"+cfbf_value[cnt]);
			}
			 
			cnt++;
		}
		 
		
		this.m_useTime = (Utils.getCurrenttime() - time_start)/(double)1000;
		this.m_numRed = Utils.booleanSelectedNum(newB);
		//System.out.println(this.algname+"Success!");
		return newB;
	}
	public String getInformation(){
		String str = this.algname+"->所用时间:" + Utils.doubleFormat("0.0000", this.m_useTime)+"s  约简个数："+this.m_numRed+"\n"+this.m_process;
		str += "最终约简:"+Arrays.toString(this.m_selectAtt);
		System.out.println(str);
		return str;
		}

	public static void main(String[] args) throws Exception {
		String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/wine.arff";
		Instances m_data = new Instances(new FileReader(fn));
		m_data.setClassIndex(m_data.numAttributes()-1); 
		SimilarityStyle sstyle = new SStyle_MaxMin();
		
		// Replace missing values   //被均值代替
		ReplaceMissingValues m_ReplaceMissingValues = new ReplaceMissingValues();
		m_ReplaceMissingValues.setInputFormat(m_data);
		m_data = Filter.useFilter(m_data, m_ReplaceMissingValues);
		
		
		ImplicatorTnormStyle itstyle = new ITStyle_Lukasiewicz(); 
		MStyle_FCBF mg = new MStyle_FCBF(m_data, false, 10, -1);
		//MStyle_FCBF mg = new MStyle_FCBF(m_data, true, -1, -1);
		mg.getInformation();
	}
}
