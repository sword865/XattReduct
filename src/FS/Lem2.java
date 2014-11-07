package FS;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Vector;

import helpLib.Utils_entropy;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class Lem2 extends Classifier {
	
	class Descriptors implements Comparable{
		int attno;
		double value;
		Vector<Integer> datas;

		Descriptors(int attno, double value, Vector<Integer> datas) {
			this.attno = attno;
			this.value = value;
			this.datas = datas;
		}

		public Descriptors() {
			// TODO Auto-generated constructor stub
		}
		
		public int compareTo(Object o1){
			if(this.attno > ((Descriptors)o1).attno || 
				(this.attno==((Descriptors)o1).attno&&
				 this.value > ((Descriptors)o1).value))
				return 1;
			else
				return -1;
		}
	}

	static Instances FinalRuleSet=null;
	double[] ruleconf=null;
	double[] rulecovered=null;
	final static double UNKNOW=Double.NaN;
	
	public static boolean[][] getupapproximation(Instances dataset,
		boolean[][] AttrEC, boolean[][] deciAttrEC) {
		boolean[][] UpperAppro = new boolean[deciAttrEC.length][deciAttrEC[0].length];
		for (int i = 0; i < UpperAppro.length; i++) {
			Arrays.fill(UpperAppro[i], false);
		}
		for (int i = 0; i < deciAttrEC[0].length; i++) {
			for (int j = 0; j < AttrEC[0].length; j++) {
				boolean flag = false;
				for (int k = 0; k < AttrEC.length; k++) {
					if (deciAttrEC[k][i] && AttrEC[k][j]) {
						flag = true;
						break;
					}
				}
				if (flag) {
					for (int k = 0; k < AttrEC.length; k++) {
						if (AttrEC[k][j])
							UpperAppro[k][i] = true;
					}
				}
			}
		}

		return UpperAppro;
	}

	private static final long serialVersionUID = 1L;

	@Override
	public void buildClassifier(Instances dataset) throws Exception {
		// TODO Auto-generated method stub
		int att = dataset.numAttributes();
		boolean[] D = new boolean[att];
		Arrays.fill(D, false);
		D[att - 1] = true;
		boolean[][] deciAttrEC = Utils_entropy.getEquivalenceClass(dataset, D);
		double[] decisionvalue=new double [deciAttrEC[0].length];
		for(int i=0;i<deciAttrEC[0].length;i++){
			for(int j=0;j<deciAttrEC.length;j++){
				if(deciAttrEC[j][i]){
					decisionvalue[i]=dataset.instance(j).classValue();
					break;
				}
			}
		}
		boolean[] B = new boolean[att];
		Arrays.fill(B, true);
		B[att - 1] = false;
		boolean[][] allAttrEC = Utils_entropy.getEquivalenceClass(dataset, B);

		boolean[][] UpperAppro = getupapproximation(dataset, allAttrEC,
				deciAttrEC);

		Vector<Descriptors> all_single_descriptor = new Vector<Descriptors>();

		for (int i = 0; i < dataset.numAttributes()-1; i++) {
			boolean[] A = new boolean[att];
			Arrays.fill(A, false);
			A[i] = true;
			boolean[][] AttrEC = Utils_entropy.getEquivalenceClass(dataset, A);
			for (int j = 0; j < AttrEC[0].length; j++) {
				Vector<Integer> tmpdata = new Vector<Integer>();
				for (int k = 0; k < AttrEC.length; k++) {
					if (AttrEC[k][j])
						tmpdata.add(k);
				}
				if (tmpdata.size() > 0) {
					Descriptors tmp = new Descriptors(i, dataset.instance(
							tmpdata.elementAt(0)).value(i), tmpdata);
					all_single_descriptor.add(tmp);
				}
			}
		}
		Descriptors[] tmpdes=new Descriptors[all_single_descriptor.size()];
		for(int i=0;i<tmpdes.length;i++){
			tmpdes[i]=all_single_descriptor.elementAt(i);
		}
		Arrays.sort(tmpdes);
		all_single_descriptor.clear();
		for(int i=0;i<tmpdes.length;i++){
			all_single_descriptor.add(tmpdes[i]);
		}
		FinalRuleSet = new Instances(dataset, 0);
		for (int i = 0; i < UpperAppro[0].length; i++) {
			boolean[] X = new boolean[UpperAppro.length];
			boolean flag=false;
			for (int j = 0; j < UpperAppro.length; j++){
				X[j] = UpperAppro[j][i];
				if(!flag&&X[j]){
					flag=true;
				}
			}
			if(flag){
				Instances RuleSetDec = Getfinalrules(dataset, all_single_descriptor, X,decisionvalue[i]);
				for (int j = 0; j < RuleSetDec.numInstances(); j++) {
					FinalRuleSet.add(RuleSetDec.instance(j));
				}
			}
		}
		
		//求规则置信度
		ruleconf=new double[FinalRuleSet.numInstances()];
		rulecovered=new double[FinalRuleSet.numInstances()];
		for (int k = 0; k < FinalRuleSet.numInstances(); k++)
		{		
			 double[] tmp= RuleConfidence(FinalRuleSet.instance(k), dataset);
			 ruleconf[k]=tmp[0];
			 rulecovered[k]=tmp[1];
		}

	};
	
	boolean RelationBetweenRulesAttr(Instance rule,Instance data){
		boolean contain=true;
		for(int i=0;i<rule.numAttributes()-1;i++){
			if((!rule.isMissing(i)) && rule.value(i)!=data.value(i)){
				contain=false;
				break;
			}
		}
		return contain;
	}
	
	double[] RuleConfidence(Instance rule,Instances dataset){
		double covered=0;
		double correct=0;
		for(int i=0;i<dataset.numInstances();i++){
			boolean rela = RelationBetweenRulesAttr(rule, dataset.instance(i));
			if (rela)
			{
				covered ++;
				if(rule.classValue()==dataset.instance(i).classValue()){
					correct++;
				}
			}
		}
		double[] rsl=new double[2];
		rsl[0]=0;rsl[1]=0;
		if(covered>0){
			rsl[0]=correct/covered;
			rsl[1]=covered;
			return rsl;
		}
		else{
			System.out.printf("what's wrong with your brain?, this rule covers nothing!!!");
			return rsl;
		}
	}

	boolean setEqual(Vector<Integer> s1, Vector<Integer> s2) {
		if (s1.size() != s2.size()) {
			return false;
		} else {
			for (int i = 0; i < s1.size(); i++) {
				if (s1.elementAt(i).intValue() != s2.elementAt(i).intValue()) {
					return false;
				}
			}
			return true;
		}
	}

	Vector<Integer> setConjuction(Vector<Integer> s1, Vector<Integer> s2) {
		Vector<Integer> rsl = new Vector<Integer>();
		if (s1.size() == 0) {
			rsl = s2;
			return rsl;
		} else if (s2.size() == 0) {
			rsl = s1;
			return rsl;
		}

		int ind1, ind2;
		for (ind1 = 0, ind2 = 0; ind1 < s1.size() && ind2 < s2.size();) {
			if (s1.elementAt(ind1).intValue() == s2.elementAt(ind2).intValue()) {
				rsl.add(s1.elementAt(ind1));
				ind1++;
				ind2++;
			} else if (s1.elementAt(ind1).intValue() < s2.elementAt(ind2).intValue()) {
				rsl.add(s1.elementAt(ind1));
				ind1++;
			} else {
				rsl.add(s2.elementAt(ind2));
				ind2++;
			}
		}

		while (ind1 < s1.size()) {
			rsl.add(s1.elementAt(ind1));
			ind1++;
		}
		while (ind2 < s2.size()) {
			rsl.add(s2.elementAt(ind2));
			ind2++;
		}
		return rsl;
	}

	Vector<Integer> setSubtract(Vector<Integer> s1, Vector<Integer> s2) {
		Vector<Integer> rsl = new Vector<Integer>();
		if (s2.size() == 0) {
			rsl = s1;// 未测试vector赋值
			return rsl;
		} else if (s1.size() == 0) {
			return rsl;
		}
		int ind1, ind2;
		for (ind1 = 0, ind2 = 0; ind1 < s1.size() && ind2 < s2.size();) {
			if (s1.elementAt(ind1).intValue() == s2.elementAt(ind2).intValue()) {
				ind1++;
				ind2++;
			} else if (s1.elementAt(ind1).intValue() < s2.elementAt(ind2).intValue()) {
				rsl.add(s1.elementAt(ind1));
				ind1++;
			} else {
				ind2++;
			}
		}
		while (ind1 < s1.size()) {
			rsl.add(s1.elementAt(ind1));
			ind1++;
		}
		return rsl;
	}

	Vector<Integer> setIntersection(Vector<Integer> s1, Vector<Integer> s2) {
		// s1 and s1 are sorted
		Vector<Integer> re = new Vector<Integer>();
		int ind1=0, ind2=0;
		for (ind1 = 0, ind2 = 0; ind1 < s1.size() && ind2 < s2.size();) {
			int t1=s1.elementAt(ind1);
			int t2=s2.elementAt(ind2);
			if (s1.elementAt(ind1).intValue() == s2.elementAt(ind2).intValue()) {
				re.add(s1.elementAt(ind1));
				ind1++;
				ind2++;
			} else if (s1.elementAt(ind1).intValue() < s2.elementAt(ind2).intValue()) {
				ind1++;
			} else {
				ind2++;
			}
		}
		return re;
	}

	public Instances Getfinalrules(Instances dataset,
			Vector<Descriptors> all_single_descriptor_sorted, boolean[] X,
			double decision) throws Exception {
		Instances RuleSet = new Instances(dataset, 0);

		Vector<Integer> G = new Vector<Integer>();
		Vector<Integer> DUP = new Vector<Integer>();
		for (int i = 0; i < X.length; i++) {
			if (X[i]) {
				G.add(i);
				DUP.add(i);
			}
		}

		Vector<Vector<Descriptors>> TT = new Vector<Vector<Descriptors>>();
		Descriptors BestDes = new Descriptors();
		int pos = -1;
		Vector<Integer> TTCoverObj = new Vector<Integer>();
		
		while (G.size() > 0) {
			Vector<Descriptors> T = new Vector<Descriptors>();
			Vector<Integer> TCoverObj = new Vector<Integer>();
			Vector<Integer> TBInt = new Vector<Integer>();
			Vector<Descriptors> TG = (Vector<Descriptors>)all_single_descriptor_sorted.clone();
			while ((T.size() == 0) || (!setEqual(TBInt, TCoverObj))) {
				int MaxIntSize = -1;
				int MinSize = dataset.numInstances() + 100000;
				for (int i = 0; i < TG.size(); i++) {
					int tmpsize = TG.elementAt(i).datas.size();
					// 求描述子i与G的交集
					Vector<Integer> intsec = setIntersection(
							TG.elementAt(i).datas, G);
					int intsize = intsec.size();
					// 选择交集最大的描述子，若有相同的选覆盖对象少的描述子，若相同，选第一个
					if ((intsize > MaxIntSize)
							|| ((intsize == MaxIntSize) && (tmpsize < MinSize))) {
						BestDes = TG.elementAt(i);
						pos = i;
						MaxIntSize = intsize;
						MinSize = TG.elementAt(i).datas.size();
					}
				}
				T.add(BestDes);
				Vector<Integer> tmp = setIntersection(BestDes.datas, G);
				G = tmp;
				if (pos > -1)
					TG.removeElementAt(pos);
				// 计算T覆盖的对象集合TCoverObj
				TCoverObj = T.elementAt(0).datas;
				for (int j = 1; j < T.size(); j++) {
					Vector<Integer> tmp1 = setIntersection(TCoverObj,
							T.elementAt(j).datas);
					TCoverObj = tmp1;
				}
				TBInt = setIntersection(TCoverObj, DUP);
			}
			// 对T进行约简
			while (T.size() > 1) {
				int i = 0;
				// 检测从T中删除描述子i得到的T1覆盖的对象是否超出了DUP的范围
				for (; i < T.size(); i++) {
					//
					Vector<Descriptors> T1;
					T1 = (Vector<Descriptors>) T.clone();
					T1.removeElementAt(i);
					Vector<Integer> tmpint = T1.elementAt(0).datas;
					for (int j = 1; j < T1.size(); j++) {
						Vector<Integer> tmp1 = setIntersection(tmpint,
								T1.elementAt(j).datas);
						tmpint = tmp1;
					}
					Vector<Integer> tmp1 = setIntersection(tmpint, DUP);
					if (setEqual(tmp1, tmpint)) {
						T = T1;
						break;
					}
				}
				if (i == T.size())
					break;
			}
			TT.add(T);
			// 计算T覆盖的对象集合TCoverObj
			TCoverObj = T.elementAt(0).datas;
			for (int j = 1; j < T.size(); j++) {
				Vector<Integer> tmp1 = setIntersection(TCoverObj,
						T.elementAt(j).datas);
				TCoverObj = tmp1;
			}

			// 求TT覆盖的对象
			Vector<Integer> tmp1 = setConjuction(TTCoverObj, TCoverObj);
			TTCoverObj = tmp1;
			G = setSubtract(DUP, TTCoverObj);
		}
		// 对TT约简
		while (TT.size() > 1) {
			int i = 0;
			// 检测从TT中删除描述子集合i后得到TT1，TT1覆盖对象是否是DUP
			for (; i < TT.size(); i++) {
				Vector<Vector<Descriptors>> TT1;
				TT1 = (Vector<Vector<Descriptors>>) TT.clone();
				TT1.removeElementAt(i);

				// 计算TT1覆盖的对象集合
				Vector<Integer> tmp = new Vector<Integer>();
				for (int j = 0; j < TT1.size(); j++) {
					Vector<Integer> tmp1 = TT1.elementAt(j).elementAt(0).datas;
					for (int k = 1; k < TT1.elementAt(j).size(); k++) {
						Vector<Integer> tmp2 = setIntersection(tmp1, TT1
								.elementAt(j).elementAt(k).datas);
						tmp1 = tmp2;
					}
					Vector<Integer> tmp2 = setConjuction(tmp, tmp1);
					tmp = tmp2;
				}
				if (setEqual(tmp, DUP)) {
					TT = TT1;
					break;
				}
			}
			if (i == TT.size())
				break;
		}
		// 将描述子用规则表示出来
		// Instance tt;

		for (int i = 0; i < TT.size(); i++) {
			Instance tmp = new Instance(dataset.numAttributes());
			for (int j = 0; j < TT.elementAt(i).size(); j++) {
				tmp.setValue(TT.elementAt(i).elementAt(j).attno, TT
						.elementAt(i).elementAt(j).value);
			}
			tmp.setValue(tmp.numAttributes() - 1, decision);
			RuleSet.add(tmp);
		}
		return RuleSet;
	}

	public double classifyInstance(Instance instance) throws Exception {
		double rsl = -1;
		double maxconf=-1;
		double curselcovered=-1;
		int rulesel=-1;
		for(int i=0;i<FinalRuleSet.numInstances();i++){
			boolean rela = RelationBetweenRulesAttr(FinalRuleSet.instance(i),instance);
			if(rela && ruleconf[i]>maxconf){
				maxconf=ruleconf[i];
				rulesel=i;
				curselcovered=rulecovered[i];
			}else if(rela && ruleconf[i]==maxconf && curselcovered<rulecovered[i]){
				maxconf=ruleconf[i];
				rulesel=i;
				curselcovered=rulecovered[i];				
			}
			
		}
		rsl=UNKNOW;
		if(rulesel!=-1){
			rsl=FinalRuleSet.instance(rulesel).classValue();
		}else{
			//System.out.printf("Now, we get an uncovered instance!\n");
		}
		
		return rsl;
	}
	
	public static void main(String[] args) throws Exception {
		File f = new File("D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\zoo_changed1.arff");
		// set data begin
		Instances traindataset = new Instances(new FileReader(f));
		traindataset.setClassIndex(traindataset.numAttributes()-1);
		f = new File("D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\zoo_changed2.arff");
		// set data begin
		Instances testdataset = new Instances(new FileReader(f));
		testdataset.setClassIndex(testdataset.numAttributes()-1);
		
		Lem2 classifier = new Lem2();
		classifier.buildClassifier(traindataset);
		for(int i=0;i<classifier.ruleconf.length;i++){
			System.out.printf(classifier.ruleconf[i]+" ");
		}
		int missed = 0;
		int uncovered = 0;
		float total = 0;
		float rate = 0;
		System.out.printf("\n");
		for (int i = 0; i < testdataset.numInstances() ; i++)
		{
			double des=classifier.classifyInstance(testdataset.instance(i));
			if (Double.isNaN(des)) 
			{
				uncovered ++;
			}else if (des != testdataset.instance(i).classValue())
			{
				missed++;
				System.out.printf(i+":"+des+"\n");
				//cout << i << ":" << rsl << endl;
			}	
		}

		total = missed + uncovered;
		rate=1-total/testdataset.numInstances();
		
		return;
	}

}
