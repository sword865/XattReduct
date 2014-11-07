package helpLib;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;

import helpLib.Utils_fuzzy.xFuzzySimilarity;
import helpLib.oneAlgorithm.xStyle;

//import Xreducer_core.Utils_fuzzy.xFuzzySimilarity;
//import Xreducer_struct.oneAlgorithm.xStyle;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class Utils {
	 
	static public double[] multicrossValidateModel(Classifier classifier,
            Instances data, int numRun ,int numFolds) throws Exception{

		    double[] res = new double [numRun];
		    for(int i=0; i<numRun; ++i ){	    	
		    	Evaluation eval = new Evaluation(data);
	    	    eval.setPriors(data);
		    	eval.crossValidateModel(classifier, data, numFolds, new Random(i));
		    	res[i]=1.0-eval.errorRate();
		    }
		    return res;
	}
	static public double onecrossValidateModel(Classifier classifier,
            Instances data, int numFolds, int randomI) throws Exception{ 	
		    	Evaluation eval = new Evaluation(data);
	    	    eval.setPriors(data);
		    	eval.crossValidateModel(classifier, data, numFolds, new Random(randomI));
		    	return 1.0-eval.errorRate();

	}
    static private double evaluateMethod_AC(Classifier c, Instances dataset, int fold, int seed) throws Exception{
        Evaluation eval = new Evaluation(dataset);
        eval.setPriors(dataset);
        //System.out.println(eval.toSummaryString(true));
        eval.crossValidateModel(c, dataset, fold, new Random(seed));
        //System.out.println(eval.toSummaryString(true));
        double d = eval.errorRate();
        return 1.0-d;
    }
  //从boolean[]转换到int[] remove;
	public static int[] boolean2remove(boolean[] resB){
		int cnt = 0;
		for (int i=0;i<resB.length-1;++i){
			if(resB[i]){
				cnt++;
			}
		}
		int[] remove = new int[resB.length-1-cnt];
		int j = 0;
		for(int i=0;i<resB.length-1;++i){
			if(!resB[i]){
				remove[j++]=i;
			}	
		}
		return remove;
	}
	public static int[] boolean2select(boolean[] resB) {
		// TODO Auto-generated method stub
		int cnt = 0;
		for (int i=0;i<resB.length-1;++i){
			if(resB[i]){
				cnt++;
			}
		}
		int[] select = new int[cnt];
		int j = 0;
		for(int i=0;i<resB.length-1;++i){
			if(resB[i]){
				select[j++]=i;
			}	
		}
		return select;
	}
	public static int[] boolean2select_wDec(boolean[] resB) {
		// TODO Auto-generated method stub
		int cnt = 0;
		for (int i=0;i<resB.length;++i){
			if(resB[i]){
				cnt++;
			}
		}
		int[] select = new int[cnt];
		int j = 0;
		for(int i=0;i<resB.length;++i){
			if(resB[i]){
				select[j++]=i;
			}	
		}
		return select;
	}
	public static boolean[] Instances2DecBoolean(Instances dataset){
		int N = dataset.numAttributes();
		boolean[] D = new boolean[N];
		D[N-1] = true;
		return D;
	}
	public static boolean[] Instances2FullBoolean(Instances dataset){
		int N = dataset.numAttributes();
		boolean[] A = new boolean[N];
		Arrays.fill(A, true);	
		A[N-1] = false;
		return A;
	}
	public static boolean isAllFalse(boolean[] des){
		boolean isAllFalse = true;
		for(int i=0;i<des.length;++i){
			if(des[i]){
				isAllFalse = false;
				break;
			}
		}
		return isAllFalse;
	}
	public static boolean isAllTrue(boolean[] des){
		boolean isAllTrue = true;
		for(int i=0;i<des.length-1;++i){ //不包括决策属性
			if(!des[i]){
				isAllTrue = false;
				break;
			}
		}
		return isAllTrue;
	}
	public static boolean[] boolsSubtract(boolean[] A, boolean[]B){//A-B
		boolean[] res = new boolean[A.length];
		for(int i=0;i<A.length;++i){
			if(A[i]&&!B[i])
				res[i]=true;
		}
		return res;
	}
	public static double[][] MinMatrics(double[][] A, double[][]B){//min(A,B)
		int N = A.length;
		double[][] res = new double[N][N];
		for(int i=0;i<N;++i){
			for(int j=i;j<N;++j){
				//res[i][j]=Math.min(A[i][j], B[i][j]);
				res[i][j]=Utils.fuzzyTnrom(A[i][j], B[i][j]);
				res[j][i]=res[i][j];
			}
		}
		return res;
	}

	public static boolean[] boolsAdd(boolean[] A, boolean[]B){//A+B
		boolean[] res = new boolean[A.length];
		for(int i=0;i<A.length;++i){
			if(A[i]||B[i])
				res[i]=true;
		}
		return res;
	}
	public static int booleanSelectedNum(boolean[]B){
		int cnt = 0;
		for(int i=0;i<B.length;++i){
			if(B[i])
				cnt++;
		}
		return cnt;
	}
	public static double fuzzyImplicator(double A, double B){
		//return Math.max(1-A, B);
		return Math.min(1-A+B, 1);
	}
	public static double fuzzyTnrom(double A, double B){
		//return Math.min(A, B);
		return Math.max(A+B-1, 0);
	}
	public static double DistanceMetrics(double x, double y){
		return x==y?0:1;
	}
	public static boolean[] boolean2oneEquivalence(boolean[][] X, int ith){
		boolean[] Xth = new boolean[X.length];
		for(int i=0;i<X.length;++i){
			Xth[i] = X[i][ith];
		}
		return Xth;
	}
	public static long getCurrenttime(){
		return System.currentTimeMillis();
	}
	public static String getCurrentDatatime(){
		SimpleDateFormat tempDate = new SimpleDateFormat("HH:mm:ss");
		return tempDate.format(new java.util.Date());
	}
	public static String getCurrentData(){
		Date date=new Date();
		SimpleDateFormat formatter = new SimpleDateFormat("dd-HH-mm");
		return formatter.format(date);
	}
	public static double log2(double x){
		if (x!=0)
		return Math.log(x) / Math.log((double)2);
		else return 0.0;
	}
	public static double getArraysSum (double[] Vas){
		double res = 0.0;
		for(int i=0;i<Vas.length;++i){
			res += Vas[i];
		}
		return res;
	}
	public static String doubleFormat(String str,double x){
		return new DecimalFormat(str).format(x);
	}
	public static double[] getStatisticsValue(double[] Vas){
		double[] ans = new double[6];//0.sum 1.max 2.min 3.mean 4var 5.std
		int ind = 0;
		int cnt = 0;
		ans[4]= 0;
		
		//System.out.println(Arrays.toString(Vas));
		for(int i=0;i<Vas.length;++i){ //处理第一个非nan的数据
			if(!Double.isNaN(Vas[i])){
				ans[0]=Vas[i];
				ans[1]=Vas[i];
				ans[2]=Vas[i];
				ind = i;
				cnt++;
				break;
			}
		}
		//System.out.println(ans[0]);
		 
		for(int i=ind+1;i<Vas.length;++i){
			if(!Double.isNaN(Vas[i])){
				ans[0] += Vas[i];
				ans[1] = Vas[i]>ans[1]?Vas[i]:ans[1];
				ans[2] = Vas[i]<ans[2]?Vas[i]:ans[2];
				cnt++;
			}
			 
		}
		ans[3] = ans[0]/(double)cnt;
		//ans[4] = ans[4]/(double)cnt-ans[3]*ans[3];
		for(int i=0;i<Vas.length;++i){
			if(!Double.isNaN(Vas[i])){
				ans[4] += (Vas[i]-ans[3])*(Vas[i]-ans[3]);
			}
			else{
				Vas[i] = ans[3];
			}
		}
		ans[4] = ans[4]/(double)(cnt-1);
		ans[5] = Math.sqrt(ans[4]);
		return ans;
	}
	//D,B全为空 返回-1.0
	public static double getEvaluateValue(xStyle style, Instances dataset,boolean[] D, boolean[] B){
		if((D!=null && Utils.isAllFalse(D))||Utils.isAllFalse(B))
			return -1.0;		
		double res = 0.0;
		xFuzzySimilarity xf = xFuzzySimilarity.MaxMin;
		switch(style){
		case informationEntropy:{
			res = Utils_entropy.getInformationEntorpy(dataset,B);
			break;
		}
		case conditionentropy:{
			res = -Utils_entropy.getConditionalEntorpy(dataset,D,B);
			break;
		}
		case positive_RSAR:{
			res = Utils_entropy.getGranulation(dataset,D,B);
			break;
		}
		case positive_DMRSAR:{
			res = Utils_entropy.getDistanceMeasure_pQ(dataset,D,B);
			break;
		}
		case SU:{
			res = Utils_entropy.getSU(dataset,D,B);
			break;
		}
		case CAIR:{
			res = Utils_entropy.getCAIR(dataset,D,B);
			break;
		}
		case IG:{
			res = Utils_entropy.getIG(dataset,D,B);
			break;
		}
		case fuzzyEntorpy_EFRFS:{
			res = Utils_fuzzy.getFuzzyEntropy_EFRFS(dataset, D, B,xf,4.0);
			break;
		}
			
		case fuzzyCEntorpy_FHFS:{
			res = Utils_fuzzy.getFuzzyConditionEntropy(dataset, D, B,xf,4.0);
			break;
		}
		case fuzzyset_FRFS:{
			res = Utils_fuzzy.getFuzzyDependencyDegree_ByFuzzySet(dataset, D, B, 0.2);
			break;
		}
		case fuzzyPositive_Low:{
			res = Utils_fuzzy.getFuzzyDependencyDegree(dataset,D, B,xf,4.0);
			break;
		}
		case fuzzyPositive_Boundary:{
			res = Utils_fuzzy.getFuzzyDependencyDegree_Boundary(dataset, D, B,xf, 4.0);
			break;
		}
		case fuzzySU:{
			res = Utils_fuzzy.getFuzzySU(dataset, D, B,xf,4.0);
			break;
		}
		case fuzzyIG:{
			res = Utils_fuzzy.getFuzzyIG(dataset, D, B,xf,4.0);
			break;
		}
		case ClassDependet:{
			res = Utils_entropy.ClassDependent(dataset, D, B);
			break;
		}
		default: break;
		}
		return res;
	}
	public static int[] getFullAtts_withDecAtt(Instances data){
		int N = data.numAttributes();
		int[] ans = new int[N];
		for(int i=0;i<N;++i){
			ans[i]=i;
		}
		return ans;
	}
	public static double getEvaluateValue(xStyle style, Instances dataset,boolean[] D, boolean[] B, double lambda){
		if((D!=null && Utils.isAllFalse(D))||Utils.isAllFalse(B))
			return -1.0;		
		double res = 0.0;
		xFuzzySimilarity xf = xFuzzySimilarity.MaxMin;
		switch(style){
		case informationEntropy:{
			res = Utils_entropy.getInformationEntorpy(dataset,B);
			break;
		}
		case conditionentropy:{
			res = Utils_entropy.getConditionalEntorpy(dataset,D,B);
			break;
		}
		case positive_RSAR:{
			res = Utils_entropy.getGranulation(dataset,D,B);
			break;
		}
		case positive_DMRSAR:{
			res = Utils_entropy.getDistanceMeasure_pQ(dataset,D,B);
			break;
		}
		case SU:{
			res = Utils_entropy.getSU(dataset,D,B);
			break;
		}
		case CAIR:{
			res = Utils_entropy.getCAIR(dataset,D,B);
			break;
		}
		case IG:{
			res = Utils_entropy.getIG(dataset,D,B);
			break;
		}
		case fuzzyEntorpy_EFRFS:{
			res = Utils_fuzzy.getFuzzyEntropy_EFRFS(dataset, D, B,xf,lambda);
			break;
		}
		case fuzzyCEntorpy_FHFS:{
			res = Utils_fuzzy.getFuzzyConditionEntropy(dataset, D, B,xf,lambda);
			break;
		}
		case fuzzyset_FRFS:{
			res = Utils_fuzzy.getFuzzyDependencyDegree_ByFuzzySet(dataset, D, B, 0.2);
			break;
		}
		case fuzzyPositive_Low:{
			res = Utils_fuzzy.getFuzzyDependencyDegree(dataset,D, B,xf,lambda);
			break;
		}
		case fuzzyPositive_Boundary:{
			res = Utils_fuzzy.getFuzzyDependencyDegree_Boundary(dataset, D, B,xFuzzySimilarity.MaxMin, lambda);
			break;
		}
		case fuzzySU:{
			res = Utils_fuzzy.getFuzzySU(dataset, D, B,xf, lambda);
			break;
		}
		case fuzzyIG:{
			res = Utils_fuzzy.getFuzzyIG(dataset, D, B,xf, lambda);
			break;
		}

		default: break;
		}
		return res;
	}
	public static double PosmeanOperater(double[] temppx) {
		// TODO Auto-generated method stub
		double ans = 0.0;
		Arrays.sort(temppx);
		int N = temppx.length;
		//均值
		
		double sum = 0.0;
		for(int i=0;i<N;++i){
			sum = sum + temppx[i];
		}
		ans = sum / (double)N;
		int a = Arrays.binarySearch(temppx, ans);
		if(Arrays.binarySearch(temppx, ans)<0)
		{
			//出现次数最多
			int[] cntrecorde = new int [N];
			int maxIndex = 0;
			Arrays.fill(cntrecorde, 0);
			for(int i=0;i<N;++i){
				for(int j=i+1;j<N;++j){
					if(temppx[i]==temppx[j]){
						cntrecorde[i]++;
						cntrecorde[j]++;
						if(cntrecorde[i]>cntrecorde[maxIndex]){
							maxIndex = i;
						}
					}
				}
			}
			ans = temppx[maxIndex];
		}
		return ans;
	}
	public static double MinMatricsSum(double[][] mRp, double[][] mRd, int ith,
			int jth) {
		double Xans = 0.0;
		for(int k=0;k<mRp.length;++k){
			Xans += Utils.fuzzyTnrom(mRp[ith][k], mRd[jth][k]);
		}
		return Xans;
	}
	public static int[] seletatt2removeAtt(int[] seletatts) {
		// TODO Auto-generated method stub
		int N = seletatts[seletatts.length-1]+1;
		int rN = N-seletatts.length;
		int[] ans = new int[rN];
		boolean[] B = new boolean[N];
		for(int i=0;i<seletatts.length;++i){
			B[seletatts[i]]=true;
		}
		int cnt = 0;
		for(int i=0;i<N;++i){
			if(!B[i]){
				ans[cnt]=i;
				cnt++;
			}
				
		}
		return ans;
	}
}
