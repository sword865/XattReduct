package helpLib;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.ConsistencySubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.SimpleCart;

public class oneAlgorithm {
	public enum xCategory {
		Fullset("Fullset"),
		Wekaalg("Wekaalg"),
		Roughsetalg("Roughsetalg"),
		FCBFalg("FCBFalg"),
		RSandFCBFalg("RSandFCBFalg"),
		NibbleRR("NibbleRR"),
		NONE("None");
	
		private String tag=null;
		private xCategory(String tag){
			this.tag = tag;
		}
		public String getValue(){
			return tag;
		}
		public static xCategory getxCategory(String str){
			//if(str =="Fullset"){
			if(str.equals("Fullset")){
				return xCategory.Fullset;
				}
			if(str.equals("Wekaalg")){
				return xCategory.Wekaalg;}
			if(str.equals("Roughsetalg")){
				return xCategory.Roughsetalg;}
			if(str.equals("FCBFalg")){
				return xCategory.FCBFalg;}
			if(str.equals("RSandFCBFalg")){
				return xCategory.RSandFCBFalg;}
			if(str.equals("NibbleRR")){
				return xCategory.NibbleRR;}
			else return xCategory.NONE;
		}
	}; //算法类别 
	public enum xStyle{
		informationEntropy("informationEntropy"),
		conditionentropy("conditionentropy"), 
		positive_RSAR("positive_RSAR"),
		positive_DMRSAR("positive_DMRSAR"),
		fuzzyset_FRFS("fuzzyset_FRFS"),
		fuzzyPositive_Low("fuzzyPositive_Low"),
		fuzzyPositive_Boundary("fuzzyPositive_Boundary"),
		fuzzyCEntorpy_FHFS("fuzzyCEntorpy_FHFS"),
		fuzzyEntorpy_EFRFS("fuzzyEntorpy_EFRFS"),
		fuzzySU("fuzzySU"),
		fuzzyIG("fuzzyIG"),
		SU("SU"), 
		CAIR("CAIR"), 
		IG("IG"),
		NONE("None"), 
		ClassDependet("ClassDependet");
		private String tag=null;
		private xStyle(String tag){
			this.tag = tag;
		}
		public String getValue(){
			return tag;
		}
		public static xStyle getxStyle(String str){
			if(str.equals("informationEntropy")){
				return xStyle.informationEntropy;}
			if(str.equals("conditionentropy")){
				return xStyle.conditionentropy;}
			if(str.equals("positive_RSAR")){
				return xStyle.positive_RSAR;}
			if(str.equals("positive_DMRSAR")){
				return xStyle.positive_DMRSAR;}
			if(str.equals("fuzzyset_FRFS")){
				return xStyle.fuzzyset_FRFS;}
			if(str.equals("fuzzyPositive_Low")){
				return xStyle.fuzzyPositive_Low;}
			if(str.equals("fuzzyPositive_Boundary")){
				return xStyle.fuzzyPositive_Boundary;}
			if(str.equals("fuzzyCEntorpy_FHFS")){
				return xStyle.fuzzyCEntorpy_FHFS;}
			if(str.equals("fuzzyEntorpy_EFRFS")){
				return xStyle.fuzzyEntorpy_EFRFS;}
			if(str.equals("fuzzySU")){
				return xStyle.fuzzySU;}
			if(str.equals("fuzzyIG")){
				return xStyle.fuzzyIG;}
			if(str.equals("SU")){
				return xStyle.SU;}
			if(str.equals("CAIR")){
				return xStyle.CAIR;}
			if(str.equals("IG")){
				return xStyle.IG;}
			if(str.equals("ClassDependet")){
				return xStyle.ClassDependet;}
			else return xStyle.NONE;
		}
	}; //评价类别 
	public xCategory category;
	public xStyle style;
	public int ID; //序号
	public String algname = null; //算法名称
	public double alpha = -1; //算法参数
	public ASEvaluation eval = null;
	public ASSearch search = null;
	public String evalname = null;
	public String searchname = null;
	public boolean flag = false;
	
	
	public int numReduce = 1;
	public int numRun = 10;
	public enum xTrainClassifier{ NaiveBayes("NaiveBayes");
		private String tag=null;
		private xTrainClassifier(String tag){
			this.tag = tag;
		}
		public String getValue(){
			return tag;
		}
		public static Classifier getClassifier(String str){
			//newmethod Classifier.forName(classifierString, options)
			if(str.equals("NaiveBayes"))
				return new NaiveBayes();
			if(str.equals("J48"))
				return new J48();
			if(str.equals("SimpleCart"))
				return new SimpleCart();
			if(str.equals("LibSVM"))
				return new LibSVM();
			else return null;
		}
		
	};
	public Classifier cl = new NaiveBayes();
	public String clname = "NaiveBayes";
	public int numFold = 10;
	
	
	public String startTime = "";
	public String endTime = "";
	public double redTime = 0.0; //约简时间
	public double trainTime = 0.0; //训练时间
	public int[] selectedAtt = null; //算选属性
	public double[] ACs = null;
	public static  void copy(oneAlgorithm des, oneAlgorithm temp){
		//oneAlgorithm temp = new oneAlgorithm();
		temp.style = des.style;
		temp.category = des.category;
		temp.ID = des.ID;
		temp.algname = des.algname;
		temp.alpha = des.alpha;
		temp.eval = des.eval;
		temp.search = des.search;
		temp.numReduce = des.numReduce;
		temp.numRun = des.numRun;
		temp.cl = des.cl;
		temp.numFold = des.numFold;
		
		temp.startTime = des.startTime;
		temp.endTime = des.endTime;
		temp.redTime = des.redTime;
		temp.selectedAtt = des.selectedAtt;
		temp.ACs = des.ACs;
		//return temp;
		
	}
	public static String  Search2Str( ASSearch search){
		if(search ==null)
			return null;
		String str = null;
			if(search.equals(new GreedyStepwise())){
				str = "GreedyStepwise";
			}
			if(search.equals(new Ranker())){
				str = "Ranker";
			}

		return str;
		
	}
	
	public static String  Eval2Str( ASEvaluation eval){
		if(eval ==null)
			return null;
		String str = null;
			if(eval.equals(new CfsSubsetEval())){
				
				str = "CfsSubsetEval";
			}
			if(eval.equals(new ConsistencySubsetEval())){
				str = "ConsistencySubsetEval";
			}
			if(eval.equals(new ReliefFAttributeEval())){
				str = "ReliefFAttributeEval";
			}
			if(eval.equals(new WrapperSubsetEval())){
				str = "WrapperSubsetEval";
			}
		return str;
		
	}
	public static ASEvaluation  Str2Eval( String str, Classifier cl){
		 if(str.equals("CfsSubsetEval")){
			 return new CfsSubsetEval();
		 }
		 if(str.equals("ConsistencySubsetEval")){
			 return new ConsistencySubsetEval();
		 }
		 if(str.equals("ReliefFAttributeEval")){
			 return new ReliefFAttributeEval();
		 }
		 if(str.equals("WrapperSubsetEval")){
			 WrapperSubsetEval wa = new WrapperSubsetEval();
   		  	 wa.setClassifier(cl);
			 return wa;
		 }
		 return null;
	}
	public static ASSearch  Str2Search( String str, double alpla){
		if(str.equals("GreedyStepwise")){
			return new GreedyStepwise();
		}
		if(str.equals("Ranker"))
		{
			Ranker se = new Ranker();
			se.setThreshold(alpla);
			return se;
		}
		else return null;
	}
}
