package Xreducer_fuzzy;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Arrays;

import org.apache.commons.math.stat.descriptive.rank.Max;

import helpLib.Utils;
import helpLib.Utils_entropy;
import helpLib.Utils_fuzzy.xFuzzySimilarity;
import weka.core.Instances;

public class MeasureStyle {
	public SimilarityStyle m_sstyle = null;
	public ImplicatorTnormStyle m_itstyle = null;
	public Instances m_data = null;
	public String m_process = "";
	public double m_useTime = 0.0;
	public int m_numRed = 0;
	public int[] m_selectAtt = null;
	public String algname = "";
	public MeasureStyle(Instances data, SimilarityStyle sstyle, ImplicatorTnormStyle itstyle){
		this.m_data = data;
		this.m_sstyle = sstyle;
		this.m_itstyle = itstyle;
		
	}
	public double getMeausureValue(boolean[] D, boolean[] B){
		return 0;
	}
	public boolean[] getReduceAtt(){
		return null;
	}
	public int[] getSelectedAtt(){
		boolean[] red = getReduceAtt();
		red[red.length-1]=true;
		SimpleDateFormat tempDate = new SimpleDateFormat("HH:mm:ss");
		String datetime = tempDate.format(new java.util.Date());
		 
		System.out.println(this.algname+"Success!+At "+datetime);
		return Utils.boolean2select_wDec(red);
	}
	public String getInformation(){
		return null;
	}
}
