package net.imglib2.atlas.color;

import java.awt.event.ActionEvent;
import java.util.Random;
import java.util.stream.IntStream;

import org.scijava.ui.behaviour.util.AbstractNamedAction;

import bdv.viewer.ViewerPanel;

public class UpdateColormap extends AbstractNamedAction
{

	public static int alpha( final float alpha )
	{
		assert alpha >= 0.0 && alpha <= 1.0;
		final int val = Math.round( alpha * 255 );
		return val << 24;
	}

	public static int getMask( final float alpha )
	{
		return alpha( alpha ) | 255 << 16 | 255 << 8 | 255 << 0;
	}

	private final ColorMapColorProvider colorProvider;

	private final int nLabels;

	private final ViewerPanel viewer;


	public UpdateColormap(final ColorMapColorProvider colorProvider, final int nLabels, final ViewerPanel viewer, final float alpha)
	{
		super( "Update Color Map" );
		this.colorProvider = colorProvider;
		this.nLabels = nLabels;
		this.viewer = viewer;
	}

	public void updateColormap()
	{
		colorProvider.setColors(IntStream.range( 0, nLabels ).toArray() );
	}

	@Override
	public void actionPerformed( final ActionEvent arg0 )
	{
		updateColormap();
		viewer.requestRepaint();
	}

}
