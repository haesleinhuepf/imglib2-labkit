
package net.imglib2.labkit.models;

import net.imagej.ImgPlus;
import net.imagej.axis.Axes;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.labkit.inputimage.ImgPlusViewsOld;
import net.imglib2.labkit.inputimage.InputImage;
import net.imglib2.labkit.labeling.Labeling;
import net.imglib2.labkit.segmentation.Segmenter;
import net.imglib2.labkit.segmentation.weka.TimeSeriesSegmenter;
import net.imglib2.labkit.segmentation.weka.TrainableSegmentationSegmenter;
import net.imglib2.labkit.utils.LabkitUtils;
import net.imglib2.labkit.utils.Notifier;
import net.imglib2.labkit.utils.progress.SwingProgressWriter;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.labkit.utils.DimensionUtils;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.ValuePair;
import org.scijava.Context;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Serves as a model for PredictionLayer and TrainClassifierAction
 */
public class DefaultSegmentationModel implements SegmentationModel,
	SegmenterListModel<SegmentationItem>
{

	private final Context context;
	private final InputImage inputImage;
	private final ImageLabelingModel imageLabelingModel;
	private final Holder<SegmentationItem> selectedSegmenter;
	private final List<SegmentationItem> segmenters = new ArrayList<>();
	private final RandomAccessibleInterval<?> compatibleImage;
	private final CellGrid grid;
	private final Holder<Boolean> segmentationVisibility = new DefaultHolder<>(
		true);
	private final Notifier listeners = new Notifier();
	private final BiFunction<Context, InputImage, Segmenter> segmenterFactory;

	public DefaultSegmentationModel(InputImage inputImage, Context context) {
		this(inputImage, context, TrainableSegmentationSegmenter::new);
	}

	public DefaultSegmentationModel(InputImage inputImage, Context context,
		BiFunction<Context, InputImage, Segmenter> segmenterFactory)
	{
		this.context = context;
		this.inputImage = inputImage;
		this.segmenterFactory = segmenterFactory;
		ImgPlus<? extends NumericType<?>> image = inputImage.imageForSegmentation();
		ImgPlus<? extends NumericType<?>> intervalWithoutChannels = ImgPlusViewsOld.hyperSlice(image,
			Axes.CHANNEL, 0);
		Labeling labeling = Labeling.createEmpty(Arrays.asList("background",
			"foreground"), intervalWithoutChannels);
		this.imageLabelingModel = new ImageLabelingModel(inputImage.showable(),
			labeling, ImgPlusViewsOld.hasAxis(image, Axes.TIME), inputImage.getDefaultLabelingFilename());
		this.compatibleImage = image;
		this.grid = LabkitUtils.suggestGrid(image);
		this.selectedSegmenter = new DefaultHolder<>(addSegmenter());
	}

	private Segmenter initClassifier() {
		Segmenter segmenter = segmenterFactory.apply(context, inputImage);
		return ImgPlusViewsOld.hasAxis(inputImage.imageForSegmentation(), Axes.TIME)
			? new TimeSeriesSegmenter(segmenter) : segmenter;
	}

	public Context context() {
		return context;
	}

	@Override
	public ImageLabelingModel imageLabelingModel() {
		return imageLabelingModel;
	}

	@Override
	public Labeling labeling() {
		return imageLabelingModel.labeling().get();
	}

	@Override
	public RandomAccessibleInterval<?> image() {
		return compatibleImage;
	}

	@Override
	public CellGrid grid() {
		return grid;
	}

	@Override
	public List<SegmentationItem> segmenters() {
		return Collections.unmodifiableList(segmenters);
	}

	@Override
	public Holder<SegmentationItem> selectedSegmenter() {
		return selectedSegmenter;
	}

	@Override
	public AffineTransform3D labelTransformation() {
		return imageLabelingModel.labelTransformation();
	}

	@Override
	public SegmentationItem addSegmenter() {
		SegmentationItem segmentationItem = new SegmentationItem(this,
			initClassifier());
		segmenters.add(segmentationItem);
		listeners.notifyListeners();
		return segmentationItem;
	}

	@Override
	public void train(SegmentationItem item) {
		SwingProgressWriter progressWriter = new SwingProgressWriter(null,
			"Training in Progress");
		progressWriter.setVisible(true);
		progressWriter.setProgressBarVisible(false);
		progressWriter.setDetailsVisible(false);
		try {
			item.train(Collections.singletonList(new ValuePair<>(image(),
				labeling())));
		}
		catch (CancellationException e) {
			progressWriter.setVisible(false);
			JOptionPane.showMessageDialog(null, e.getMessage(), "Training Cancelled",
				JOptionPane.PLAIN_MESSAGE);
		}
		catch (Throwable e) {
			progressWriter.setVisible(false);
			JOptionPane.showMessageDialog(null, e.toString(), "Training Failed",
				JOptionPane.WARNING_MESSAGE);
		}
		finally {
			progressWriter.setVisible(false);
		}
	}

	@Override
	public void remove(SegmentationItem item) {
		if (segmenters.size() <= 1) return;
		segmenters.remove(item);
		if (!segmenters.contains(selectedSegmenter.get())) selectedSegmenter.set(
			segmenters.get(0));
		listeners.notifyListeners();
	}

	@Override
	public void trainSegmenter() {
		train(selectedSegmenter().get());
	}

	@Override
	public Holder<Boolean> segmentationVisibility() {
		return segmentationVisibility;
	}

	@Override
	public Notifier listChangeListeners() {
		return listeners;
	}

	public <T extends IntegerType<T> & NativeType<T>>
		List<RandomAccessibleInterval<T>> getSegmentations(T type)
	{
		RandomAccessibleInterval<?> image = image();
		Stream<Segmenter> trainedSegmenters = getTrainedSegmenters();
		return trainedSegmenters.map(segmenter -> {
			RandomAccessibleInterval<T> labels = new CellImgFactory<>(type).create(
				image);
			segmenter.segment(image, labels);
			return labels;
		}).collect(Collectors.toList());
	}

	public List<RandomAccessibleInterval<FloatType>> getPredictions() {
		RandomAccessibleInterval<?> image = image();
		Stream<Segmenter> trainedSegmenters = getTrainedSegmenters();
		return trainedSegmenters.map(segmenter -> {
			int numberOfClasses = segmenter.classNames().size();
			RandomAccessibleInterval<FloatType> prediction = new CellImgFactory<>(
				new FloatType()).create(DimensionUtils.appendDimensionToInterval(image,
					0, numberOfClasses - 1));
			segmenter.predict(image, prediction);
			return prediction;
		}).collect(Collectors.toList());
	}

	public boolean isTrained() {
		return getTrainedSegmenters().findAny().isPresent();
	}

	private Stream<Segmenter> getTrainedSegmenters() {
		return segmenters().stream().filter(Segmenter::isTrained).map(x -> x);
	}

}
