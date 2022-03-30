# -*- coding: utf-8 -*-

from functools import partial
import click

from ms_imputer import fit_nmf

click.option = partial(click.option, show_default=True)

settings = dict(help_option_names=['-h', '--help'])

@click.group(options_metavar='',
             subcommand_metavar='<command>',
             context_settings=settings)
def cli():
    """
    Impute missing data with non-negative matrix factorization
    """
    pass

cli.add_command(fit_nmf, name='fit')

if __name__ == "__main__":
    cli()