package helpLib;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;
import java.util.Vector;

import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math.stat.descriptive.moment.Variance;
import org.apache.commons.math.stat.descriptive.rank.Max;
import org.apache.commons.math.stat.descriptive.rank.Min;
import org.apache.commons.math.stat.descriptive.summary.Product;
import org.apache.commons.math.stat.descriptive.summary.Sum;

//import Xreducer_fuzzy.SStyle_MaxMin;
//import Xreducer_fuzzy.SimilarityStyle;
 

import weka.core.Instances;

public class Utils_fuzzy {

	public enum xFuzzySimilarity{
		Line("Line"),
		Gaussian("Gaussian"),
		MaxMin("MaxMin"),
		Abs1lambda("Abs1lambda"),
		NONE("None");
		
		private String tag=null;
		private xFuzzySimilarity(String tag){
			this.tag = tag;
		}
		public String getValue(){
			return tag;
		}
		public static xFuzzySimilarity xFuzzySimilarity(String str){
			if(str.equals("Line")){
				return xFuzzySimilarity.Line;
				}
			if(str.equals("Gaussian")){
				return xFuzzySimilarity.Gaussian;}
			if(str.equals("MaxMin")){
				return xFuzzySimilarity.MaxMin;}
			if(str.equals("Abs1lambda")){
				return xFuzzySimilarity.Abs1lambda;}
			else return xFuzzySimilarity.NONE;
		}
	}
	public static double[][] getFuzzySimilarityRelation_Nominal(Instances dataset, int seletAtt){
		int N = dataset.numInstances();
		double[][] MR = new double[N][N];
		for(int i=0;i<N;++i){
			MR[i][i]=1.0;
			for(int j=i+1;j<N;++j){
				
				//System.out.println(dataset.instance(i).value(seletAtt));
				//System.out.println(dataset.instance(j).value(seletAtt));
				MR[i][j] = dataset.instance(i).value(seletAtt)==dataset.instance(j).value(seletAtt)?1:0; //miss值的处理没有做
				MR[j][i] = MR[i][j];
			}
		}
		return MR;
		}
	public static double[][] getFuzzySimilarityRelation_Numeric(Instances dataset, int seletAtt,xFuzzySimilarity xfs){
		return getFuzzySimilarityRelation_Numeric(dataset,seletAtt,xfs,4.0);	
	}
	
	public static double[][] getFuzzySimilarityRelation_Numeric(Instances dataset, int seletAtt,xFuzzySimilarity xfs, double lambda){
		int N = dataset.numInstances();
		int M = dataset.numAttributes();
		double[][] MR = new double[N][N];
		double[] Vas = new double[N];
		for(int i=0;i<N;++i){
			Vas[i] = dataset.instance(i).value(seletAtt);
		}
		/*double Vmax = new Max().evaluate(Vas);
		double Vmin = new Min().evaluate(Vas);
		double Vmean = (Vmax - Vmin)/2.0;
		for(int i=0;i<N;++i){
			if(Double.isNaN(Vas[i]))
				Vas[i] = Vmean;
		}
		double Vvar = new Variance().evaluate(Vas);
		double Vstdvar = new StandardDeviation().evaluate(Vas);*/
		//System.out.println(new Sum().evaluate(Vas)+":"+Vmax+":"+Vmin+":"+ new Mean().evaluate(Vas)+":"+Vvar+":"+Vstdvar);
		//double[] Vas = dataset.attributeToDoubleArray(seletAtt);
		double[] Vs = Utils.getStatisticsValue(Vas);
		double Vmax = Vs[1]; // 最大
		double Vmin = Vs[2]; // 最小
		double Vvar = Vs[4];  // 方差
		double Vstdvar = Vs[5];// 标准方差
		//Utils.getStatisticsValue(Vas);
		//System.out.println(Arrays.toString(Vs));
		if(Vmax==Vmin){
			for(int i=0;i<N;++i){
				Arrays.fill(MR[i], 1.0);
			}
			return MR;
		}
		switch(xfs){
		case Line:{
			for(int i=0;i<N;++i){
				MR[i][i]=1.0;
				for(int j=i+1;j<N;++j){
					MR[i][j] = 1-Math.abs((Vas[i]-Vas[j])/(Vmax-Vmin));
					MR[j][i] = MR[i][j];
				}
			}
			break;
		}
		case Gaussian:{
			for(int i=0;i<N;++i){
				MR[i][i]=1.0;
				for(int j=i+1;j<N;++j){
					MR[i][j] = Math.exp(-(Vas[i]-Vas[j])*(Vas[i]-Vas[j])/Vvar); 
					MR[j][i] = MR[i][j];
				}
			}
			break;
		}
		case MaxMin:{
			for(int i=0;i<N;++i){
				MR[i][i]=1.0;
				for(int j=i+1;j<N;++j){
					double part1 = (Vas[j]-(Vas[i]-Vstdvar))/(Vas[i]-(Vas[i]-Vstdvar));
					double part2 = ((Vas[i]+Vstdvar)-Vas[j])/((Vas[i]+Vstdvar)-Vas[i]);
					//MR[i][j] = new Double(new DecimalFormat( ".000" ).format( Math.max(Math.min(part1, part2), 0) ) );

					MR[i][j] = Math.max(Math.min(part1, part2), 0);
					MR[j][i] = MR[i][j];
				}
			}
			break;
		}
		case Abs1lambda:{
			for(int i=0;i<N;++i){
				MR[i][i]=1.0;
				for(int j=i+1;j<N;++j){
					//double abs = Math.abs((Vas[i]-Vas[j])/Vmax);
					double abs = Math.abs((Vas[i]-Vas[j])/(Vmax-Vmin));
					if(abs<=1.0/lambda)
						MR[i][j] = 1-lambda*abs; 
					else MR[i][j] = 0.0;
					MR[j][i] = MR[i][j];
				}
			}
			break;
		}
		default:break;
		}
		return MR;
	}
	public static double getFuzzySimilarityRelation_Numeric_Quick(xFuzzySimilarity xfs,double Vasi,double Vasj, double Vmax,double Vmin,double Vstdvar ){
		double ans = 0.0;
		switch(xfs){
		case Line:{
			ans = 1-Math.abs((Vasi-Vasj)/(Vmax-Vmin));
			break;
		}
		case Gaussian:{
			ans = Math.exp(-(Vasi-Vasj)*(Vasi-Vasj)/Vstdvar*Vstdvar); 
			break;
		}
		case MaxMin:{
			double part1 = (Vasj-(Vasi-Vstdvar))/(Vasi-(Vasi-Vstdvar));
			double part2 = ((Vasi+Vstdvar)-Vasj)/((Vasi+Vstdvar)-Vasi);
			//MR[i][j] = new Double(new DecimalFormat( ".000" ).format( Math.max(Math.min(part1, part2), 0) ) );
			ans = Math.max(Math.min(part1, part2), 0);
			break;
		}
		case Abs1lambda:{
			//double abs = Math.abs((Vasi-Vasj)/Vmax);
			double abs = Math.abs((Vasi-Vasj)/(Vmax-Vmin));
			if(abs<=1.0/4.0)
				ans = 1-4.0*abs; 
			else ans = 0.0;
			break;
		}
		default:break;
		}
		return ans;
	}
	public static double[][] getFuzzySimilarityRelation(Instances dataset,boolean[] P,xFuzzySimilarity xfs,double lambda){
		int N = dataset.numInstances();
		int M = dataset.numAttributes();
		double[][] newMR = new double[N][N];
		for(int i=0;i<N;++i){
			Arrays.fill(newMR[i], 1);
		}
		for(int j=0;j<M;++j){
			double[][] tempMR =  new double[N][N];
			if(P[j]){
				if(dataset.attribute(j).isNumeric()){
					tempMR = getFuzzySimilarityRelation_Numeric(dataset,j,xfs,lambda);
				}
				else
					tempMR = getFuzzySimilarityRelation_Nominal(dataset,j);
				//if(tempMR[1][6]!=0.0)
					//System.out.println(j);
				
				newMR = Utils.MinMatrics(newMR, tempMR);
				//if(newMR[1][6]!=0.0)
					//System.out.println(j+":"+newMR[1][6]);
			}			
		}
		return newMR;
	}
	public static double getFuzzyInformationEntropy(Instances dataset,boolean[] P,xFuzzySimilarity xfs,double lambda){
		double ans = 0.0;
		double[][] MRp = getFuzzySimilarityRelation(dataset,P,xfs,lambda);
		int N = dataset.numInstances();
		for(int i=0;i<N;++i){
			//System.out.println(Arrays.toString(MRp[i]));
			double part1 = new Sum().evaluate(MRp[i])/(double)N;
			//System.out.println(new Sum().evaluate(MRp[i]));
			if(part1!=1)
			ans = ans + Math.log(part1) / Math.log((double)2);
		}
		return -ans/(double)N;
	}
	public static double getFuzzyInformationEntropy_Quick(Instances dataset,boolean[] P,xFuzzySimilarity xfs,double lambda){
		double ans = 0.0;
		int U = dataset.numInstances();
		int N = dataset.numAttributes();
		for(int i=0;i<U;++i){
			double[] sumV = new double[U];
			Arrays.fill(sumV, 1);
			for(int k=0;k<N;++k){
				if(P[k]&&dataset.attribute(k).isNumeric()){
					double[] Vas = dataset.attributeToDoubleArray(k);			
					double[] vasStatistics = Utils.getStatisticsValue(Vas);
					for(int q=0;q<U;++q) {
						sumV[q] = Utils.fuzzyTnrom(sumV[q],getFuzzySimilarityRelation_Numeric_Quick(
								xFuzzySimilarity.MaxMin,
								dataset.instance(i).value(k),dataset.instance(q).value(k),
								vasStatistics[1],vasStatistics[2],vasStatistics[5]));
					}
				}
				else if(P[k]&&!dataset.attribute(k).isNumeric()){
					for(int q=0;q<U;++q)
						sumV[q] = Utils.fuzzyTnrom(sumV[q],dataset.instance(i).value(k)==dataset.instance(q).value(k)?1:0);
				} 
			}
			ans = ans + Utils.log2(Utils.getArraysSum(sumV)/(double)U);
		}
		return -ans/(double)U;
	}
	public static double getFuzzyEntropy_EFRFS(Instances dataset,boolean[] D, boolean[] P,xFuzzySimilarity xfs,double lambda){
		double ans = 0.0;
		double[][] MRp = getFuzzySimilarityRelation(dataset,P,xfs,lambda);
		double[][] MRd = getFuzzySimilarityRelation(dataset,D,xfs,lambda);
		//double[][] MRd = getFuzzySimilarityRelation_Nominal(dataset,dataset.classIndex());
		int N = dataset.numInstances();
		double Fall = 0;
		for(int i=0;i<N;++i){		
			double Fi = new Sum().evaluate(MRp[i]);
			double Hfi = 0.0;
			for(int j=0;j<N;++j){
				double FDij = Utils.MinMatricsSum(MRp,MRd,i,j);
				double Dj = new Sum().evaluate(MRd[j]);
				//System.out.println(i+":"+j+":"+(FDij));
				if(FDij!=0&&(Fi!=0&&Dj!=0))		
					Hfi = Hfi-(Math.log(FDij/Fi) / Math.log((double)2))*(FDij/Fi)/Dj;
			}
			ans = ans + Fi*Hfi;
			Fall = Fall + Fi;
			//System.out.println(Arrays.toString(MRpd[i]));
			//System.out.println(ans);
			
		}
		
		if(ans==0)
			return 1.0;
		else	return 1-ans/Fall;	
	}
	public static double getFuzzyConditionEntropy(Instances dataset,boolean[] D, boolean[] P,xFuzzySimilarity xfs,double lambda){
		double ans = 0.0;
		double[][] MRp = getFuzzySimilarityRelation(dataset,P,xfs,lambda);
		double[][] MRpd = Utils.MinMatrics(MRp,getFuzzySimilarityRelation_Nominal(dataset,dataset.classIndex()));
		int N = dataset.numInstances();
		for(int i=0;i<N;++i){
			double part1 = new Sum().evaluate(MRpd[i]);
			double part2 = new Sum().evaluate(MRp[i]);
			//System.out.println(Arrays.toString(MRpd[i]));
			//System.out.println(part1+"/"+part2+"="+part1/part2);
			if(part1!=0)		
				ans = ans+Math.log(part1/part2) / Math.log((double)2);
		}
		//System.out.println( MRpd[514][79]);
		//System.out.println( getFuzzySimilarityRelation(dataset,D,xfs,lambda)[514][79]);
		//System.out.println( dataset.instance(514).classValue());
		//System.out.println( dataset.instance(79).classValue());
		if(ans==0)
			return 0.0;
		else	return -ans/(double)N;	
	}
	public static double getFuzzyConditionEntropy_Quick(Instances dataset,boolean[] D, boolean[] P,xFuzzySimilarity xfs,double lambda){
		double ans = 0.0;
		int U = dataset.numInstances();
		int N = dataset.numAttributes();
		int classind = dataset.classIndex();
		for(int i=0;i<U;++i){
			double[] sumV = new double[U];
			Arrays.fill(sumV, 1);
			for(int k=0;k<N;++k){
				if(P[k]&&dataset.attribute(k).isNumeric()){
					double[] Vas = dataset.attributeToDoubleArray(k);			
					double[] vasStatistics = Utils.getStatisticsValue(Vas);
					for(int q=0;q<U;++q) {
						sumV[q] = Utils.fuzzyTnrom(sumV[q],getFuzzySimilarityRelation_Numeric_Quick(
								xfs,
								dataset.instance(i).value(k),dataset.instance(q).value(k),
								vasStatistics[1],vasStatistics[2],vasStatistics[5]));
					}
				}
				else if(P[k]&&!dataset.attribute(k).isNumeric()){
					for(int q=0;q<U;++q)
						sumV[q] = Utils.fuzzyTnrom(sumV[q],dataset.instance(i).value(k)==dataset.instance(q).value(k)?1:0);
				} 
			}
			double part2 = Utils.getArraysSum(sumV);
			for(int q=0;q<U;++q)
				sumV[q] = Utils.fuzzyTnrom(sumV[q],dataset.instance(i).value(classind)==dataset.instance(q).value(classind)?1:0);
			double part1 = Utils.getArraysSum(sumV);
			System.out.println(part1+"/"+part2+"="+part1/part2);
			if(part1!=0)		
				ans = ans+Utils.log2(part1/part2);
		}
		//System.out.println(ans);	
		if(ans==0)
			return 0.0;
		else	return -ans/(double)U;	
	}
	public static double getFuzzyIG(Instances dataset, boolean[] D, boolean[] P,xFuzzySimilarity xfs,double lambda){
		double fHd = getFuzzyInformationEntropy(dataset, D,xfs,lambda);
		double fHdp = getFuzzyConditionEntropy(dataset, D, P, xfs,lambda);
		return fHd-fHdp;
	}
	public static double getFuzzySU(Instances dataset, boolean[] D, boolean[] P,xFuzzySimilarity xfs,double lambda){
		double fHp = getFuzzyInformationEntropy(dataset, P,xfs,lambda);
		double fHd = getFuzzyInformationEntropy(dataset, D,xfs,lambda);
		double fHdp = getFuzzyConditionEntropy(dataset, D, P, xfs,lambda);
	//System.out.println(fHd+":"+fHp+":"+fHdp);
		double ans = 2.0*(fHd-fHdp)/(fHd+fHp);
		if(ans<0.0000000001)
			return 0.0;
		return ans;
	}
	public static double getFuzzySU_Quick(Instances dataset, boolean[] D, boolean[] P,xFuzzySimilarity xfs,double lambda){
		/*double fHp = 0.0;
		double fHd = 0.0;
		double fHdp = 0.0;
		int U = dataset.numInstances();
		int N = dataset.numAttributes();
		int classind = dataset.classIndex();
		
		double[][] sumV = new double[U][U];
		for(int i=0;i<U;++i)
			Arrays.fill(sumV[i], 1);
		double[] sumD = new double[U];
		Arrays.fill(sumD, 1);
			for(int k=0;k<N;++k){
				if(P[k]&&dataset.attribute(k).isNumeric()){
					double[] Vas = dataset.attributeToDoubleArray(k);			
					double[] vasStatistics = Utils.getStatisticsValue(Vas);
					for(int i=0;i<U;++i)
						for(int q=0;q<U;++q) 
							sumV[i][q] = Utils.fuzzyTnrom(sumV[i][q],getFuzzySimilarityRelation_Numeric_Quick(
									xFuzzySimilarity.MaxMin,
									dataset.instance(i).value(k),dataset.instance(q).value(k),
									vasStatistics[1],vasStatistics[2],vasStatistics[5]));
				}
				else if(P[k]&&!dataset.attribute(k).isNumeric()){
					for(int i=0;i<U;++i)
					  for(int q=0;q<U;++q)
						sumV[i][q] = Utils.fuzzyTnrom(sumV[i][q],dataset.instance(i).value(k)==dataset.instance(q).value(k)?1:0);
				}
			}
			for(int i=0;i<U;++i){
				double partp = Utils.getArraysSum(sumV[i]);
				for(int q=0;q<U;++q){
				sumD[q] = dataset.instance(i).value(classind)==dataset.instance(q).value(classind)?1:0;
				sumV[i][q] = Utils.fuzzyTnrom(sumV[i][q],sumD[q]);
				}
				double partpd = Utils.getArraysSum(sumV[i]);
				double partd = Utils.getArraysSum(sumD);
				fHd += Utils.log2(partd/(double)U);
				fHp += Utils.log2(partp/(double)U);
				if(partp!=0)
					fHdp += Utils.log2(partpd/partp);
			}
		
		//System.out.println(ans);	
		double ans = 2.0*(fHd-fHdp)/(fHd+fHp);
		if(ans<0.0000000001)
			return 0.0;
		return ans;*/
		
		double fHp = 0.0;
		double fHd = 0.0;
		double fHdp = 0.0;
		int U = dataset.numInstances();
		int N = dataset.numAttributes();
		int classind = dataset.classIndex();
		for(int i=0;i<U;++i){
			double[] sumV = new double[U];
			Arrays.fill(sumV, 1);
			double[] sumD = new double[U];
			Arrays.fill(sumD, 1);
			for(int k=0;k<N;++k){
				if(P[k]&&dataset.attribute(k).isNumeric()){
					double[] Vas = dataset.attributeToDoubleArray(k);			
					double[] vasStatistics = Utils.getStatisticsValue(Vas);
					for(int q=0;q<U;++q) {
						sumV[q] = Utils.fuzzyTnrom(sumV[q],getFuzzySimilarityRelation_Numeric_Quick(
								xFuzzySimilarity.MaxMin,
								dataset.instance(i).value(k),dataset.instance(q).value(k),
								vasStatistics[1],vasStatistics[2],vasStatistics[5]));
					}
				}
				else if(P[k]&&!dataset.attribute(k).isNumeric()){
					for(int q=0;q<U;++q)
						sumV[q] = Utils.fuzzyTnrom(sumV[q],dataset.instance(i).value(k)==dataset.instance(q).value(k)?1:0);
				} 
			}
			double partp = Utils.getArraysSum(sumV);
			for(int q=0;q<U;++q){
				sumD[q] = dataset.instance(i).value(classind)==dataset.instance(q).value(classind)?1:0;
				sumV[q] = Utils.fuzzyTnrom(sumV[q],sumD[q]);
				}
			double partpd = Utils.getArraysSum(sumV);
			double partd = Utils.getArraysSum(sumD);
				fHd += Utils.log2(partd/(double)U);
				fHp += Utils.log2(partp/(double)U);
			if(partp!=0)
				fHdp += Utils.log2(partpd/partp);
		}
		//System.out.println(ans);	
		double ans = 2.0*(fHd-fHdp)/(fHd+fHp);
		if(ans<0.0000000001)
			return 0.0;
		return ans;
	}
	public static double getFuzzyDependencyDegree_Quick(Instances dataset,boolean[] D,boolean[] P,xFuzzySimilarity xfs,double lambda){
		double ans = 0.0;
		//int N = dataset.numInstances();
		int U = dataset.numInstances();
		int N = dataset.numAttributes();
		///double[][] MRp = getFuzzySimilarityRelation(dataset,P,xfs,lambda);
		boolean[][] MRd = Utils_entropy.getEquivalenceClass(dataset,D);
	
		int dM = MRd[0].length;
		double[][] vasStatistics = new double[Utils.booleanSelectedNum(P)][8];
		int vsIndex = 0;
		for(int k=0;k<N;++k){	
			if(P[k]&&dataset.attribute(k).isNumeric()){
				vasStatistics[vsIndex][0] = 0;
				vasStatistics[vsIndex][1] = k;
				double[] Vas = dataset.attributeToDoubleArray(k);			
				double[] temp = Utils.getStatisticsValue(Vas);
				vasStatistics[vsIndex][2] = temp[0];
				vasStatistics[vsIndex][3] = temp[1];
				vasStatistics[vsIndex][4] = temp[2];
				vasStatistics[vsIndex][5] = temp[3];
				vasStatistics[vsIndex][6] = temp[4];
				vasStatistics[vsIndex][7] = temp[5];
				vsIndex++;
			}
			else if(P[k]&&!dataset.attribute(k).isNumeric()){			
				vasStatistics[vsIndex][0] = 1;
				vasStatistics[vsIndex][1] = k;
				vsIndex++;
			}
		}
		//SimilarityStyle xfx =  new SStyle_MaxMin();
		for(int i=0;i<U;++i){
			double[] Upos = new double[dM];
			Arrays.fill(Upos, 100);
			double[] sumV = new double[U];
			Arrays.fill(sumV, 1);
					
			for(int q=0;q<U;++q) {
				for(int k=0;k<vasStatistics.length;++k){
				if(vasStatistics[k][0]==0){
					int pIndex = (int)vasStatistics[k][1];
					
					//xfx.SimilaritySetting(vasStatistics[k][3],vasStatistics[k][4],vasStatistics[k][7]);
					//double sd =xfx.getSimilarityValue(dataset.instance(i).value(pIndex),dataset.instance(q).value(pIndex));
					double sd = getFuzzySimilarityRelation_Numeric_Quick(
							xFuzzySimilarity.MaxMin,
							dataset.instance(i).value(pIndex),dataset.instance(q).value(pIndex),
							vasStatistics[k][3],vasStatistics[k][4],vasStatistics[k][7]);
					sumV[q] = Utils.fuzzyTnrom(sumV[q],sd);
				}
				else{
					int pIndex = (int)vasStatistics[k][1];
					sumV[q] = Utils.fuzzyTnrom(sumV[q],dataset.instance(i).value(pIndex)==dataset.instance(q).value(pIndex)?1:0);			
				}
				}
				for(int d=0;d<dM;++d) //对每个决策类	
					Upos[d] = Math.min(Utils.fuzzyImplicator(sumV[q],MRd[q][d]?1:0),Upos[d]);
						
			}			    				
			ans = ans + new Max().evaluate(Upos);// 最大
		}
		return ans/(double)U;
	}
	public static double getFuzzyDependencyDegree(Instances dataset,boolean[] D,boolean[] P,xFuzzySimilarity xfs,double lambda){
		double ans = 0.0;
		int N = dataset.numInstances();
		double[][] MRp = getFuzzySimilarityRelation(dataset,P,xfs,lambda);
		boolean[][] MRd = Utils_entropy.getEquivalenceClass(dataset,D);
		int dM = MRd[0].length;
		for(int i=0;i<N;++i){
			double[] Upos = new double[dM];
			for(int k=0;k<dM;++k){ //对每个决策类
				double minUx = 100.0;
				for(int j=0;j<N;++j){
					if(Utils.fuzzyImplicator(MRp[i][j],MRd[j][k]?1:0)<minUx){
						minUx = Utils.fuzzyImplicator(MRp[i][j], MRd[j][k]?1:0);
					}
				}
				Upos[k] = minUx;
			}
			ans = ans + new Max().evaluate(Upos);// 最大
		}
		
		return ans/(double)N;
	}
	public static double getFuzzyDependencyDegree_Boundary(Instances dataset,boolean[] D,boolean[] P,xFuzzySimilarity xfs,double lambda){
		double ans = 0.0;
		int N = dataset.numInstances();
		double[][] MRp = getFuzzySimilarityRelation(dataset,P,xfs,lambda);
		boolean[][] MRd = Utils_entropy.getEquivalenceClass(dataset,D);
		int dM = MRd[0].length;
		for(int i=0;i<N;++i){	
		       for(int k=0;k<dM;++k){ //对每个决策类

				
				double minUx = 100.0;
				double maxDx = -1.0;
				for(int j=0;j<N;++j){
					/*if(Utils.fuzzyImplicator(MRp[i][j],MRd[j][k]?1:0)<minUx){
						minUx = Utils.fuzzyImplicator(MRp[i][j], MRd[j][k]?1:0);
					}
					if(Utils.fuzzyTnrom(MRp[i][j],MRd[j][k]?1:0)>maxDx){
						maxDx = Utils.fuzzyTnrom(MRp[i][j],MRd[j][k]?1:0);
					}*/
					minUx = Math.min(Utils.fuzzyImplicator(MRp[i][j], MRd[j][k]?1:0),minUx);
					maxDx = Math.max(Utils.fuzzyTnrom(MRp[i][j],MRd[j][k]?1:0),maxDx);
				}
				//System.out.println(k+"&&"+(i+1)+"%%"+maxDx+"-"+minUx+"="+(maxDx-minUx));
				ans += maxDx-minUx;
			}
			//ans = ans + Upos;
		}
		//System.out.println(ans+"/"+N+"*"+dM);
		return 1-ans/(double)(N*dM);
		
	}
	public static class  FuzzyGen {
		public double[][] ab = null;
		public double lambda = 0.0;
		public int U  = 0;
		public int N = 0;
		public Instances data = null;
		public FuzzyGen(Instances dataset, double lambda){
			this.data = dataset;
			this.N = dataset.numAttributes();
			this.U = dataset.numInstances();
			this.ab = new double[N-1][4];  //不算决策属性
			this.lambda = lambda;
			for(int i = 0;i<N-1;++i){
				double[] oneA = dataset.attributeToDoubleArray(i);
				double mean = new Mean().evaluate(oneA);
				double std = new StandardDeviation().evaluate(oneA);
				ab[i][0]=mean-std;
				ab[i][1]=mean-lambda*std;
				ab[i][2]=mean+lambda*std;
				ab[i][3]=mean+std;
			}
		}
		public Vector<double[][]>  getFuzzy(){
			Vector<double[][]> Fi = new Vector<double[][]>();
			for(int k=0;k<N-1;++k){
				double[][] oneAtt = new double[U][3];
				for(int i=0;i<U;++i){
					double x = this.data.instance(i).value(k);
					double X1 = (ab[k][1]-x)/(ab[k][1]-ab[k][0]);
					double X2 = (x-ab[k][0])/(ab[k][1]-ab[k][0]);
					double Y1 = (ab[k][3]-x)/(ab[k][3]-ab[k][2]);
					double Y2 = (x-ab[k][2])/(ab[k][3]-ab[k][2]);
					oneAtt[i][0] = Math.max(Math.min(X1, 1),0);
					oneAtt[i][1] = Math.max(Math.min(Math.min(X2, Y1), 1), 0);
					oneAtt[i][2] = Math.max(Math.min(Y2, 1),0);
				}
				Fi.add(oneAtt);
			}
			boolean[][] oneDecb = Utils_entropy.getEquivalenceClass(data, Utils.Instances2DecBoolean(data)); //决策属性 crips
			double[][] oneDec = new double[oneDecb.length][oneDecb[0].length];
			for(int i=0;i<oneDecb.length;++i){
				for(int j=0;j<oneDecb[0].length;++j){
					if(oneDecb[i][j])
						oneDec[i][j] = 1.0;
					else
						oneDec[i][j] = 0.0;
				}
			}
			Fi.add(oneDec);
			return Fi;
		}
	}
	public static double getFuzzyDependencyDegree_ByFuzzySet(Instances dataset,boolean[] D, boolean[] P, double lambda){
		FuzzyGen fg = new FuzzyGen(dataset,lambda);
		Vector<double[][]> Fi = fg.getFuzzy();
		double ans = 0.0;		
		int Qindex =  Utils.boolean2select(D).length!=0? Utils.boolean2select(D)[0]:D.length-1;
		int[] Ps = Utils.boolean2select(P);
		int Pnum = Utils.booleanSelectedNum(P);
		int N = Fi.get(0).length;
		int Unum = 1;
		int[][] Unums = new int[Pnum][2];
		for(int i=0;i<Pnum;++i){
			Unum = Unum * Fi.get(Ps[i])[0].length;
			Unums[i][1] = Fi.get(Ps[i])[0].length;
		}
		Unums[0][0] = 1;
		for(int i=1;i<Pnum;++i){
			Unums[i][0]=Unums[i-1][1]*Unums[i-1][0];
		}
		int[][] Ups = new int[Unum][Pnum];
		for(int i=0;i<Pnum;++i){
			for(int j=0;j<Unum;++j){
				Ups[j][i]=(j/Unums[i][0])%Unums[i][1];
			}
		}
		Vector<double[]> Us =new Vector<double[]>();
		for(int i=0;i<N;++i){
			double[] Utemp = new double[Unum];
			for(int j=0;j<Unum;++j){
				double Pmin = Fi.get(Ps[0])[i][Ups[j][0]];
				for(int k=1;k<Pnum;++k)
					if(Fi.get(Ps[k])[i][Ups[j][k]]<Pmin)
						Pmin = Fi.get(Ps[k])[i][Ups[j][k]];
				Utemp[j] = Pmin;
			}
			Us.add(Utemp);
		}	
		for(int i=0;i<N;++i){
			double fmax = -1.0;
			for(int d=0;d<Fi.get(Qindex)[0].length;++d){
				for(int f = 0;f<Unum;++f){				
					double ftemp = 0.0;
					double ftempmin = 100000; 
					for(int j=0;j<N;++j){				
						ftemp = Math.max(1-Us.get(j)[f], Fi.get(Qindex)[j][d]);
						//System.out.println(Us.get(j)[f]+"%%"+Fi.get(Qindex)[j][d]+"***"+ftemp);
						if(ftemp<ftempmin)
							ftempmin = ftemp;
					}
					
					if(Math.min(Us.get(i)[f],ftempmin) > fmax)
						fmax = Math.min(Us.get(i)[f],ftempmin);
				}
			}
			//System.out.println(fmax);
			ans = ans + fmax;
		}
		//System.out.println(ans);
		return ans/(double)N;
	}
	/**
	 * @param args
	 * @throws IOException 
	 * @throws Exception 
	 */
	public static void randomdataset (String fn, int ins, int att, Random rnd) throws IOException{
		String filein = "@RELATION ex \n\n @attribute a REAL\n@attribute b REAL\n@attribute c " +
				"REAL\n @attribute d REAL\n@attribute q {0,1}\n\n@data\n\n";

        File f = new File(fn);
        f.delete();
        int[] d = {1,0,1,1,1,0};
        //Random rnd = new Random(seed);
        RandomAccessFile mm =null;
        try {
            mm = new RandomAccessFile(fn, "rw");
            for(int i=0;i<ins;++i){
            	for(int k=0;k<att;++k){
            		String v = new DecimalFormat( "0.00" ).format(rnd.nextDouble());
            		filein += v +" ";
            	}
            	filein += Integer.toString(d[i])+"\n";
            }
            mm.writeBytes(filein);

        } catch (IOException e1) {
            // TODO 自动生成 catch 块
            e1.printStackTrace();
        } finally {
            if (mm != null) {
                try {
                    mm.close();
                } catch (IOException e2) {
                    // TODO 自动生成 catch 块
                    e2.printStackTrace();
                }
            }
        }
	}
	public static void randomdataset2 (String fn, int ins, int att,int cls, Random rnd) throws IOException{
		String filein = "@RELATION ex \n\n " +
				"@attribute n1 {0,1,2}\n" +
				"@attribute n2 {0,1,2}\n" +
				"@attribute r1 REAL\n" +
				"@attribute r2 REAL\n" +
				"@attribute r3 REAL\n" +
				"@attribute r4 REAL\n" +
				"@attribute r5 REAL\n" +
				"@attribute r6 REAL\n" +
				"@attribute r7 REAL\n" +
				"@attribute r8 REAL\n" +
				"@attribute d {0,1,2}\n\n" +
				"@data\n\n";

        File f = new File(fn);
        f.delete();
        int[] d = new int[ins];
        for(int i=0;i<ins;++i){
        	d[i]=rnd.nextInt(cls);
        }
        //Random rnd = new Random(seed);
        RandomAccessFile mm =null;
        try {
            mm = new RandomAccessFile(fn, "rw");
            for(int i=0;i<ins;++i){
               	for(int k=0;k<2;++k){
            		String v = Integer.toString(rnd.nextInt(3));
            		filein += v +" ";
            	}         	
            	for(int k=2;k<att;++k){
            		String v = new DecimalFormat( "0.00" ).format(rnd.nextDouble());
            		filein += v +" ";
            	}
            	filein += Integer.toString(d[i])+"\n";
            }
            mm.writeBytes(filein);

        } catch (IOException e1) {
            // TODO 自动生成 catch 块
            e1.printStackTrace();
        } finally {
            if (mm != null) {
                try {
                    mm.close();
                } catch (IOException e2) {
                    // TODO 自动生成 catch 块
                    e2.printStackTrace();
                }
            }
        }
	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/fuzzy/ex.arff";
		 Random rnd = new Random(4);
		int cnt = 100000;
		double lambda = 4.0;
		for(int i=0;i<cnt;++i){
			randomdataset (fn,6,4,rnd);
			wekaDiscretizeMethod dm = new wekaDiscretizeMethod(new File(fn),true);
			int U = dm.getOriginalData().numInstances();
			int N = dm.getOriginalData().numAttributes();
			boolean[] D=Utils.Instances2DecBoolean(dm.getOriginalData());
			boolean[] P=new boolean[N];			
			double[] sus = new double[N-1];
			for(int k=0;k<N-1;++k){
				P[k]=true;
				sus[k] = Utils_fuzzy.getFuzzySU(dm.getOriginalData(),D,P,xFuzzySimilarity.MaxMin,4.0);
				P[k]=false;
			}
			int[] rankIndex = new int[N-1];
			for(int k=0;k<N-1;++k){
				rankIndex [k] = k;
			}
			double[] restemp = sus.clone();
			double temp;
			int tempindex;
			for(int ii=0;ii<N-1;++ii){//冒泡法排序 
				for(int ji=0;ji< N-ii-2;++ji){
					if(restemp[ji]<restemp[ji+1]) {
						//交换restemp
						temp = restemp[ji];
						restemp[ji] = restemp[ji + 1];
						restemp[ji + 1] = temp;
						//交换entropyRankindex
						tempindex = rankIndex[ji];
						rankIndex[ji] = rankIndex[ji + 1];
						rankIndex[ji + 1] = tempindex;
					}
				}
			}
			System.out.println(i);
			boolean[] P1=new boolean[N];
			boolean[] P2=new boolean[N];	
			boolean[] P3=new boolean[N];
			P1[rankIndex[0]]=true;
			P2[rankIndex[1]]=true;
			P3[rankIndex[2]]=true;
			boolean flag1 = Utils_fuzzy.getFuzzySU(dm.getOriginalData(),P1,P2,xFuzzySimilarity.MaxMin,4.0)>=lambda*Utils_fuzzy.getFuzzySU(dm.getOriginalData(),D,P2,xFuzzySimilarity.MaxMin,4.0);
			if(!flag1)
				continue;
			boolean flag2 = Utils_fuzzy.getFuzzySU(dm.getOriginalData(),P1,P3,xFuzzySimilarity.MaxMin,4.0)<lambda*Utils_fuzzy.getFuzzySU(dm.getOriginalData(),D,P3,xFuzzySimilarity.MaxMin,4.0);
			if(!flag2)
				continue;
			P3[rankIndex[0]]=true;
			boolean flag3 = Utils_fuzzy.getFuzzyConditionEntropy(dm.getOriginalData(), D, P3, xFuzzySimilarity.MaxMin,4.0)==0;
			if(flag3)
			{
				 System.out.println(Utils_fuzzy.getFuzzySU(dm.getOriginalData(),P1,P2,xFuzzySimilarity.MaxMin,4.0)+":"+Utils_fuzzy.getFuzzySU(dm.getOriginalData(),D,P2,xFuzzySimilarity.MaxMin,4.0));
				System.out.println(Arrays.toString(rankIndex));
				System.out.println(Arrays.toString(restemp));
				break;
			}
			
		} 
		//String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/fuzzy/fuzzy-ex.arff";
		//String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/fuzzy/heart-c.arff";
	//String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/fuzzy/ex.arff";
		//String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/Data/wine.arff";
		/*Instances data = new Instances(new FileReader(fn));
		data.setClassIndex(data.numAttributes()-1);
		boolean[] D=Utils.Instances2DecBoolean(data);
		boolean[] P = new boolean[data.numAttributes()];
		boolean[] P2 = new boolean[data.numAttributes()];
		//P[1]=true;
		//P[2]=true;
		P[2]=true;
		P2[1]=true;
		//P[2]=true; P[3]=true;P[4]=true;P[6]=true;P[7]=true; P[9]=true; P[11]=true;P[12]=true;
		//P[20]=true;
		//boolean[] P = Utils.Instances2FullBoolean(data);
		
		//boolean[] P = new boolean[data.numAttributes()];
		///P[22] = true;
		//double ans = Utils_fuzzy.getFuzzyConditionEntropy(data,D,P,xFuzzySimilarity.MaxMin,4.0);
		//System.out.println(ans);
		//P[13] = false;
		double ans = Utils_fuzzy.getFuzzySU(data,P2,P,xFuzzySimilarity.MaxMin,4.0);
		//double ans2 = Utils_fuzzy.getFuzzySU(data,P2,P,xFuzzySimilarity.MaxMin,4.0);
		System.out.println(ans+":");
		
		/*String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/fuzzy/heart-c.arff";
		Instances data = new Instances(new FileReader(fn));
		data.setClassIndex(data.numAttributes()-1);
		//System.out.println(data.numClasses());
		boolean[] D=Utils.Instances2DecBoolean(data);
		boolean[] P = new boolean[data.numAttributes()];
		wekaDiscretizeMethod dm = new wekaDiscretizeMethod(new File(fn),true);
		P[0]=true;
		P[1]=true;
		P[2]=true;
		P[3]=true;
		long time = Utils.getCurrenttime();
		double ans = 0.0;
		for(int i=0;i<50;++i){
			ans = Utils_fuzzy.getFuzzyDependencyDegree_Quick(dm.getOriginalData(),D,P,xFuzzySimilarity.MaxMin,4.0);		 
		}
		time = Utils.getCurrenttime()-time;
		System.out.println(time+":"+ans);
		time = Utils.getCurrenttime();
		for(int i=0;i<50;++i){
			ans =  Utils_fuzzy.getFuzzyDependencyDegree(dm.getOriginalData(),D,P,xFuzzySimilarity.MaxMin,4.0);		 
		}
		time = Utils.getCurrenttime()-time;
		System.out.println(time+":"+ans);
			*/
		/*boolean[] Ps = new boolean[4];
		boolean[] Ds = new boolean[4];
		//Ps[0]=true;
		Ps[1]=true;
		//Ps[2]=true;
		Ds[3]=true;
		Vector<double[][]> U = new Vector<double[][]>();
		double[][] A = {{0.3,0.7,0.0},{1.0,0.0,0.0},{0.0,0.3,0.7,},{0.8,0.2,0.0},
						{0.5,0.5,0.0},{0.0,0.2,0.8},{1.0,0.0,0.0},{0.1,0.8,0.1},{0.3,0.7,0.0}};
		double[][] B = {{0.2,0.7,0.1},{1.0,0.0,0.0},{0.0,0.7,0.3},{0.0,0.7,0.3},
						{1.0,0.0,0.0},{0.0,1.0,0.0},{0.7,0.3,0.0},{0.0,0.9,0.1},{0.9,0.1,0.0}};
		double[][] C = {{0.3,0.7},{0.7,0.3},{0.6,0.4},{0.2,0.8},
						{0.0,1.0},{0.0,1.0},{0.2,0.8},{0.7,0.3},{1.0,0.0}};
		double[][] D = {{0.1,0.9,0.0},{0.8,0.2,0.0},{0.0,0.2,0.8},{0.6,0.3,0.1},
						{0.6,0.8,0.0},{0.0,0.7,0.3},{0.7,0.4,0.0},{0.0,0.0,1.0},{0.0,0.0,1.0}};
		U.add(A);U.add(B);U.add(C);U.add(D);
		double ans  = getFuzzyDependencyDegree_ByFuzzySet(U,Ds,Ps);
		System.out.println(ans+"####");*/
	}

}
