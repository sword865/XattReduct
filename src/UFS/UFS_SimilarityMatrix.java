package UFS;

import java.util.Arrays;

import helpLib.Utils;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
//import FCBFandRS.FSmethod;

public class UFS_SimilarityMatrix extends FSmethod {

	public double[][] SMatrix = null; //分辨度矩阵
	public UFS_SimilarityMatrix(Instances data,int bin, int evalutionIndex) {
		super(data);
		// Replace missing values   //被均值代替
		ReplaceMissingValues m_ReplaceMissingValues = new ReplaceMissingValues();
		try {
			m_ReplaceMissingValues.setInputFormat(m_data);
			m_data = Filter.useFilter(m_data, m_ReplaceMissingValues);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		//离散化
		if(bin == -1){
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
			unsd.setBins(bin);
			//unsd.setUseEqualFrequency(true); // If set to true, equal-frequency binning will be used instead of equal-width binning.
			try {
				unsd.setInputFormat(this.m_data);
				this.m_data = Filter.useFilter(this.m_data , unsd);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
		

		int N = data.numInstances();
		int M = data.numAttributes()-1;
		SMatrix = new double[M][M];
		for(int i=0;i<M;++i){
			SMatrix[i][i] = 1;
			for(int j=i+1;j<M;++j)
			{
				boolean[] A = new boolean[data.numAttributes()];
				A[i]=true;
				boolean[] B = new boolean[data.numAttributes()];
				B[j]=true;
				switch(evalutionIndex)
				{
				case 0:{
					SMatrix[i][j]=AttributeEvaluation.getSU(data, A, B);
					break;
				}
				case 1:{
					SMatrix[i][j]=AttributeEvaluation.DoubleDependencyDegree(data, A, B);
					break;
				}
				case 2:{
					SMatrix[i][j]=AttributeEvaluation.DiscriminateDegree(data, A, B);
					break;
				}
				default:break;
				}
				SMatrix[j][i]=SMatrix[i][j];
			}
			
			
		}
		this.algname = "UFS_ela"+evalutionIndex;
		this.m_selectAtt = getSelectedAtt();
		// TODO Auto-generated constructor stub
	}

	@Override
	public boolean[] getReduceAtt() {
		// TODO Auto-generated method stub
		int M = this.m_data.numAttributes();
		boolean[] newB = new boolean[M];
		int[] cfbf_index = new int[M-1]; //不包括决策属性
		double[] cfbf_value = new double[M-1];
		
		long time_start = Utils.getCurrenttime();		
		
		
		for(int k=0;k<M-1;++k){
			cfbf_value[k]=this.getMeanValue(this.SMatrix[k]);
			cfbf_index[k]=k;
		}
		//降序排序
		double temp;
		int tempindex;
		for(int i=0;i<M-1;++i){/* 冒泡法排序 */ 
			for(int j=0;j< M-i-2;++j){
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
		
		System.out.println(Arrays.toString(cfbf_value));
		newB[cfbf_index[0]]=true;
		 		 
		for(int k =1;k<M-1;k++){
			boolean isRud = false;
			for(int i=0;i<k;++i){
				if(newB[cfbf_index[i]]){
					int Y=cfbf_index[i]; //red->Y
					int X=cfbf_index[k]; //待定->X
					if(this.SMatrix[X][Y]<cfbf_value[k]){//X<->Y的分辨度 小于 X的平均分辨度 标为冗余
						isRud = true;
						break;
					}		
				}
			}
			if(!isRud){
				newB[cfbf_index[k]]=true;
				//System.out.println(Arrays.toString(Utils.boolean2select(newB))+"->"+cfbf_index[cfbf_index[k]]+":"+cfbf_value[cfbf_index[k]]);
			}
		}
		

		
		this.m_useTime = (Utils.getCurrenttime() - time_start)/(double)1000;
		this.m_numRed = Utils.booleanSelectedNum(newB);
		return newB;
	}

	private double getMeanValue(double[] ds) {
		// TODO Auto-generated method stub
		double sum = 0;
		for(int i=0;i<ds.length;++i)
		{
			sum+=ds[i];
		}
		return sum/(double)ds.length;
	}

}
