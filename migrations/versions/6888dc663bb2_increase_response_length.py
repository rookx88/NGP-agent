"""Increase response length

Revision ID: 6888dc663bb2
Revises: 69f3022e1d96
Create Date: 2025-02-03 10:47:32.229105

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6888dc663bb2'
down_revision: Union[str, None] = '69f3022e1d96'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('interactions', 'original_content',
               existing_type=sa.VARCHAR(length=280),
               type_=sa.String(length=1000),
               existing_nullable=True)
    op.alter_column('interactions', 'bot_response',
               existing_type=sa.VARCHAR(length=280),
               type_=sa.Text(),
               existing_nullable=True)
    op.create_index('ix_interactions_tweet_id', 'interactions', ['tweet_id'], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('ix_interactions_tweet_id', table_name='interactions')
    op.alter_column('interactions', 'bot_response',
               existing_type=sa.Text(),
               type_=sa.VARCHAR(length=280),
               existing_nullable=True)
    op.alter_column('interactions', 'original_content',
               existing_type=sa.String(length=1000),
               type_=sa.VARCHAR(length=280),
               existing_nullable=True)
    # ### end Alembic commands ###
