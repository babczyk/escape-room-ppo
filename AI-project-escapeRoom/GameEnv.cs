using System.Runtime.Serialization.Formatters;
using AI_project_escapeRoom; // Replace 'YourNamespace' with the actual namespace where Game1 is defined
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;
using System.Diagnostics;
using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;

class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 10000;
    private int currentStep;
    public List<int> PlayerMove;

    public double pick_the_box = 10;
    public double place_the_box_good = 20;
    public double finish_reward = 200;
    public double droping_box_bad = -2;
    public double culide_with_wall = -1;
    public double repeating_actions = -0.5;
    public double time_panalty = -0.1;
    public double max_steps_panalty = -5;

    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
        this.PlayerMove = new List<int>();
    }

    public double[] GetState()
    {
        return new double[]
        {
        game.player.Position.X / game.widthLevel,
        game.player.Position.Y / game.groundLevel,
        game.box.Position.X / game.widthLevel,
        game.box.Position.Y / game.groundLevel,
        game.button.Position.X / game.widthLevel,
        game.button.Position.Y / game.groundLevel,
        game.door.Position.X / game.widthLevel,
        game.door.Position.Y / game.groundLevel,
        game.player.heldBox != null ? 1.0 : 0.0,
        game.IsPressed ? 1.0 : 0.0,
        game.IsOpen ? 1.0 : 0.0
        };
    }

    public (double[], double, bool) Step(int action)
    {
        currentStep++;
        double reward = 0;
        bool IsDone = false;

        // Apply action
        switch (action)
        {
            case 0: game.player.Move(new Vector2(-10, 0)); break; // Move left
            case 1: game.player.Move(new Vector2(10, 0)); break;  // Move right
            case 2: if (game.player.IsGrounded) game.player.ApplyForce(new Vector2(0, -250)); game.player.IsGrounded = false; break; // Jump
            case 3: game.player.Grab(game.box); break; // Interact (e.g., pick up the box)
            case 4: game.player.DropHeldBox(); break; // Interact (e.g., drop the box)
        }

        PlayerMove.Add(action);
        ///////////////////////////////
        // Rewards for key objectives//
        ///////////////////////////////

        //pick the box
        if (game.player.heldBox != null)
        {
            reward += pick_the_box; // Reward for picking up the box
        }

        //placing the box on the button
        if (game.box.Intersects(game.button) && game.player.heldBox == null)
        {
            reward += place_the_box_good; // Reward for placing the box on the button
        }

        //exiting the room finish goal
        if (game.IsPressed && IsOutOfBounds(game.player))
        {
            reward += finish_reward; // Reward for escaping the room
            IsDone = true;
        }

        //reward for geting closer to the goals
        if (game.IsMovingToward(game.box, game.lastPlayerPosition) && game.player.heldBox == null
        || game.IsMovingToward(game.button, game.lastPlayerPosition) && game.player.heldBox != null)
        {
            reward += 0.2;
        }

        ///////////////////////////////
        // Penalties for incorrect behaviors//
        ///////////////////////////////
        //droping the box for no resone
        if (game.player.heldBox == null && game.previousBoxState == true
        && !game.box.Intersects(game.button))
        {
            reward -= droping_box_bad; // Penalty for dropping the box for no reason
        }

        //culiding with the walls (not the ground)
        if (!game.player.IsGrounded && game.player.Intersects(game.wall))
        {
            reward -= culide_with_wall; // Penalty for colliding with the walls
        }

        //repeating actions
        if (PlayerMove.Skip(PlayerMove.Count - 50).Distinct().Count() < 3)
        {
            reward -= repeating_actions;
        }

        //time penalty
        if (currentStep % 100 == 0)
        {
            reward -= time_panalty; // Penalty for taking too long
        }

        // Reset if out of bounds
        if (IsOutOfBounds(game.player) || IsOutOfBounds(game.box))
        {
            ResetPlayerAndBox();
        }
        //panelty for geting away from the goals
        if (!game.IsMovingToward(game.box, game.lastPlayerPosition) && game.player.heldBox == null
        || !game.IsMovingToward(game.button, game.lastPlayerPosition) && game.player.heldBox != null)
        {
            reward -= 0.1;
        }

        // Maximum steps penalty
        if (currentStep >= maxSteps)
        {
            reward -= max_steps_panalty; // Small penalty for exceeding maximum steps
            ResetPlayerAndBox();
            IsDone = true;
            currentStep = 0;
        }

        //Thread.Sleep(1);
        return (GetState(), reward, IsDone);
    }

    // Helper methods
    private bool IsOutOfBounds(GameObject obj)
    {
        return obj.Position.X < game.cameraPosition.X || obj.Position.X > game.cameraPosition.X + game.ScreenWidth ||
               obj.Position.Y < game.cameraPosition.Y || obj.Position.Y > game.cameraPosition.Y + game.ScreenHeight;
    }

    private void ResetPlayerAndBox()
    {
        ResetPlayerPosition();
        ResetBoxPosition();
    }

    private void ResetPlayerPosition()
    {
        game.player.Position = new Vector2(100, game.groundLevel - game.player.Size.Y);
    }

    private void ResetBoxPosition()
    {
        game.box.Position = new Vector2(200, game.groundLevel - game.box.Size.Y);
    }

    public bool IsDone()
    {
        return game.IsOpen || currentStep >= maxSteps;
    }
}